import io
import json
import os
import posixpath
import time
import uuid
from pathlib import PurePosixPath

import pandas as pd
import paramiko
from upload_to_hdfs import (
    HDFS_CLIENT_OPTIONS,
    docker_cp_to_container,
    docker_exec,
    q,
    resolve_hdfs_target,
    run,
    run_with_output,
)


SSH_HOST = os.getenv("SSH_HOST", "139.180.136.228")
SSH_USER = os.getenv("SSH_USER", "root")
SSH_PASSWORD = os.getenv("SSH_PASSWORD", "8[qXdBXt8mY)Lk3b")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH")

HDFS_BASE_DIR = os.getenv("HDFS_BASE_DIR", "/user/root/raw_data")
REMOTE_TMP = os.getenv("REMOTE_TMP", "/tmp")


class RemoteStorage:
    def __init__(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs = {
            "hostname": SSH_HOST,
            "username": SSH_USER,
        }
        if SSH_KEY_PATH:
            connect_kwargs["key_filename"] = SSH_KEY_PATH
        elif SSH_PASSWORD:
            connect_kwargs["password"] = SSH_PASSWORD

        self.ssh.connect(**connect_kwargs)
        self.sftp = self.ssh.open_sftp()

        target = resolve_hdfs_target(self.ssh)
        self.hdfs_cmd = target["cmd"]
        self.hdfs_container = target["container"]

        if self.hdfs_container:
            print(f"[REMOTE STORAGE] HDFS: {self.hdfs_cmd} inside {self.hdfs_container}")
        else:
            print(f"[REMOTE STORAGE] HDFS: {self.hdfs_cmd}")

    def close(self):
        self.sftp.close()
        self.ssh.close()

    def path(self, *parts):
        return posixpath.join(HDFS_BASE_DIR, *[str(part).strip("/\\") for part in parts])

    def exists(self, path):
        return self._hdfs(f"-test -e {q(path)}")[0] == 0

    def mkdir_parent(self, path):
        parent = str(PurePosixPath(path).parent)
        self._hdfs(f"-mkdir -p {q(parent)}")

    def read_json(self, path, default):
        if not self.exists(path):
            return default
        content = self._read_hdfs_file(path, encoding="utf-8")
        return json.loads(content)

    def write_json(self, path, data):
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        self._write_hdfs_file(path, payload.encode("utf-8"), overwrite=True)

    def append_csv(self, path, records, encoding="utf-8"):
        if not records:
            return

        write_header = not self.exists(path)
        output_encoding = encoding if write_header else "utf-8"
        buffer = io.StringIO()
        pd.DataFrame(records).to_csv(
            buffer,
            index=False,
            header=write_header,
            encoding=encoding,
        )
        self._append_hdfs_file(path, buffer.getvalue().encode(output_encoding))

    def init_csv(self, path, columns, encoding="utf-8"):
        if self.exists(path):
            return

        buffer = io.StringIO()
        pd.DataFrame(columns=columns).to_csv(buffer, index=False, encoding=encoding)
        self._write_hdfs_file(path, buffer.getvalue().encode(encoding), overwrite=False)

    def read_csv_column_as_str_set(self, path, column):
        if not self.exists(path):
            return set()

        content = self._read_hdfs_file(path, encoding="utf-8-sig")
        if not content.strip():
            return set()

        df = pd.read_csv(io.StringIO(content))
        if column not in df.columns:
            return set()
        return set(df[column].dropna().astype(str))

    def _hdfs(self, args):
        options = f"{HDFS_CLIENT_OPTIONS} " if HDFS_CLIENT_OPTIONS else ""
        command = f"{q(self.hdfs_cmd)} dfs {options}{args}"
        if self.hdfs_container:
            command = docker_exec(self.hdfs_container, command)
        return run_with_output(self.ssh, command)

    def _tmp_path(self, suffix=""):
        name = f"crawler-hdfs-{uuid.uuid4().hex}{suffix}"
        return posixpath.join(REMOTE_TMP, name)

    def _write_host_tmp(self, payload, suffix=""):
        host_tmp = self._tmp_path(suffix)
        with self.sftp.open(host_tmp, "wb") as remote_file:
            remote_file.write(payload)
        return host_tmp

    def _copy_to_hdfs_visible_tmp(self, host_tmp):
        if not self.hdfs_container:
            return host_tmp

        container_tmp = self._tmp_path(posixpath.splitext(host_tmp)[1])
        if not run(self.ssh, docker_cp_to_container(self.hdfs_container, host_tmp, container_tmp)):
            raise RuntimeError(f"Cannot copy temp file into container: {self.hdfs_container}")
        return container_tmp

    def _cleanup_tmp(self, host_tmp, hdfs_tmp):
        if self.hdfs_container and hdfs_tmp != host_tmp:
            run(self.ssh, docker_exec(self.hdfs_container, f"rm -f {q(hdfs_tmp)}"))
        run(self.ssh, f"rm -f {q(host_tmp)}")

    def _read_hdfs_file(self, path, encoding):
        return self._read_hdfs_bytes(path).decode(encoding)

    def _read_hdfs_bytes(self, path):
        host_tmp = self._tmp_path(posixpath.splitext(str(path))[1])
        hdfs_tmp = host_tmp

        try:
            if self.hdfs_container:
                hdfs_tmp = self._tmp_path(posixpath.splitext(str(path))[1])
                exit_code, _, stderr_text = self._hdfs(f"-get -f {q(path)} {q(hdfs_tmp)}")
                if exit_code != 0:
                    raise RuntimeError(stderr_text or f"Cannot read HDFS file: {path}")

                if not run(self.ssh, f"docker cp {q(self.hdfs_container + ':' + hdfs_tmp)} {q(host_tmp)}"):
                    raise RuntimeError(f"Cannot copy HDFS temp file from container: {path}")
            else:
                exit_code, _, stderr_text = self._hdfs(f"-get -f {q(path)} {q(host_tmp)}")
                if exit_code != 0:
                    raise RuntimeError(stderr_text or f"Cannot read HDFS file: {path}")

            with self.sftp.open(host_tmp, "rb") as remote_file:
                return remote_file.read()
        finally:
            self._cleanup_tmp(host_tmp, hdfs_tmp)

    def _write_hdfs_file(self, path, payload, overwrite):
        self.mkdir_parent(path)
        host_tmp = self._write_host_tmp(payload, posixpath.splitext(str(path))[1])
        hdfs_tmp = self._copy_to_hdfs_visible_tmp(host_tmp)
        overwrite_flag = "-f " if overwrite else ""

        try:
            exit_code, _, stderr_text = self._hdfs(
                f"-put {overwrite_flag}{q(hdfs_tmp)} {q(path)}"
            )
            if exit_code != 0:
                raise RuntimeError(stderr_text or f"Cannot write HDFS file: {path}")
        finally:
            self._cleanup_tmp(host_tmp, hdfs_tmp)

    def _append_hdfs_file(self, path, payload):
        self.mkdir_parent(path)
        host_tmp = self._write_host_tmp(payload, posixpath.splitext(str(path))[1])
        hdfs_tmp = self._copy_to_hdfs_visible_tmp(host_tmp)

        try:
            if self.exists(path):
                exit_code, stderr_text = self._append_with_retry(hdfs_tmp, path)
            else:
                exit_code, _, stderr_text = self._hdfs(
                    f"-put {q(hdfs_tmp)} {q(path)}"
                )

            if exit_code != 0:
                if self.exists(path):
                    self._rewrite_with_appended_payload(path, payload)
                else:
                    raise RuntimeError(stderr_text or f"Cannot append HDFS file: {path}")
        finally:
            self._cleanup_tmp(host_tmp, hdfs_tmp)

    def _append_with_retry(self, hdfs_tmp, path, attempts=5):
        last_stderr = ""
        for attempt in range(1, attempts + 1):
            exit_code, _, stderr_text = self._hdfs(
                f"-appendToFile {q(hdfs_tmp)} {q(path)}"
            )
            if exit_code == 0:
                return 0, ""

            last_stderr = stderr_text
            retryable = (
                "lease recovery is in progress" in stderr_text.lower()
                or "failed to replace a bad datanode" in stderr_text.lower()
            )
            if not retryable:
                return exit_code, stderr_text

            time.sleep(min(2 * attempt, 10))

        return 1, last_stderr

    def _rewrite_with_appended_payload(self, path, payload):
        existing = self._read_hdfs_bytes(path)
        self._write_hdfs_file(path, existing + payload, overwrite=True)