import os
import posixpath
import shlex

import paramiko

SSH_HOST = os.getenv("SSH_HOST", "139.180.136.228")
SSH_USER = os.getenv("SSH_USER", "root")
SSH_PASSWORD = os.getenv("SSH_PASSWORD", "8[qXdBXt8mY)Lk3b")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH")

DATA_DIR = os.getenv("DATA_DIR", "crawlers/data")
REMOTE_DATA_DIR = os.getenv(
    "REMOTE_DATA_DIR",
    "/tmp/distributed-nlp-trend-analysis/crawlers/data",
)
HDFS_BASE_DIR = os.getenv("HDFS_BASE_DIR", "/user/zett/raw_data")
HDFS_CMD = os.getenv("HDFS_CMD")
HDFS_CONTAINER = os.getenv("HDFS_CONTAINER")
HDFS_CLIENT_OPTIONS = os.getenv(
    "HDFS_CLIENT_OPTIONS",
    "-Ddfs.client.block.write.replace-datanode-on-failure.policy=NEVER "
    "-Ddfs.client.block.write.replace-datanode-on-failure.enable=false",
)

REMOTE_TMP = os.getenv("REMOTE_TMP", "/tmp")
HDFS_CANDIDATES = (
    "/opt/hadoop/bin/hdfs",
    "/usr/local/hadoop/bin/hdfs",
    "/opt/hadoop-3.3.6/bin/hdfs",
    "/opt/hadoop-3.2.1/bin/hdfs",
    "/home/hadoop/hadoop/bin/hdfs",
)


def q(value):
    return shlex.quote(str(value))


def run_with_output(ssh, command):

    stdin, stdout, stderr = ssh.exec_command(command)

    exit_code = stdout.channel.recv_exit_status()
    stdout_text = stdout.read().decode(errors="replace").strip()
    stderr_text = stderr.read().decode(errors="replace").strip()

    return exit_code, stdout_text, stderr_text


def run(ssh, command):

    exit_code, stdout_text, stderr_text = run_with_output(ssh, command)

    if exit_code != 0:
        print(f"Remote command failed: {command}")
        if stderr_text:
            print(stderr_text)

    return exit_code == 0


def docker_exec(container, command):
    return f"docker exec {q(container)} sh -lc {q(command)}"


def docker_cp_to_container(container, host_path, container_path):
    return f"docker cp {q(host_path)} {q(container + ':' + container_path)}"


def detect_hdfs_container(ssh):
    if HDFS_CONTAINER:
        return HDFS_CONTAINER

    exit_code, stdout_text, _ = run_with_output(
        ssh,
        "docker ps --format '{{.Names}}\t{{.Image}}'"
    )
    if exit_code != 0:
        return None

    candidates = []
    for line in stdout_text.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        name, image = parts
        search_text = f"{name} {image}".lower()
        if "hadoop" in search_text or "namenode" in search_text or "datanode" in search_text:
            candidates.append((name, search_text))

    for name, search_text in candidates:
        if "namenode" in search_text or "name-node" in search_text or "master" in search_text:
            return name

    for name, search_text in candidates:
        if "hadoop" in search_text:
            return name

    for name, _ in candidates:
        return name

    return None


def resolve_container_hdfs_cmd(ssh, container):
    if HDFS_CMD:
        if run(ssh, docker_exec(container, f"test -x {q(HDFS_CMD)}")):
            return HDFS_CMD
        raise RuntimeError(
            f"Cannot find executable HDFS command inside container {container}: {HDFS_CMD}. "
            "Set HDFS_CMD to the correct in-container path."
        )

    exit_code, stdout_text, _ = run_with_output(
        ssh,
        docker_exec(container, "command -v hdfs || true")
    )
    if exit_code == 0 and stdout_text:
        return stdout_text.splitlines()[0]

    for candidate in HDFS_CANDIDATES:
        if run_with_output(ssh, docker_exec(container, f"test -x {q(candidate)}"))[0] == 0:
            return candidate

    checked = ", ".join(HDFS_CANDIDATES)
    raise RuntimeError(
        f"Cannot find HDFS CLI inside container {container}. Checked PATH and: {checked}. "
        "Set HDFS_CMD to the in-container hdfs path, for example /opt/hadoop-3.2.1/bin/hdfs."
    )


def resolve_hdfs_target(ssh):
    if HDFS_CONTAINER:
        return {
            "cmd": resolve_container_hdfs_cmd(ssh, HDFS_CONTAINER),
            "container": HDFS_CONTAINER,
        }

    if HDFS_CMD:
        if run(ssh, f"test -x {q(HDFS_CMD)}"):
            return {"cmd": HDFS_CMD, "container": None}
        container = detect_hdfs_container(ssh)
        if container:
            return {
                "cmd": resolve_container_hdfs_cmd(ssh, container),
                "container": container,
            }
        raise RuntimeError(
            f"Cannot find executable HDFS command on {SSH_USER}@{SSH_HOST}: {HDFS_CMD}. "
            "Set HDFS_CMD to the correct host path, or set HDFS_CONTAINER if this path is inside Docker."
        )

    exit_code, stdout_text, _ = run_with_output(
        ssh,
        "bash -lc 'command -v hdfs || true'"
    )
    if exit_code == 0 and stdout_text:
        return {"cmd": stdout_text.splitlines()[0], "container": None}

    for candidate in HDFS_CANDIDATES:
        if run_with_output(ssh, f"test -x {q(candidate)}")[0] == 0:
            return {"cmd": candidate, "container": None}

    container = detect_hdfs_container(ssh)
    if container:
        return {
            "cmd": resolve_container_hdfs_cmd(ssh, container),
            "container": container,
        }

    checked = ", ".join(HDFS_CANDIDATES)
    raise RuntimeError(
        f"Cannot find HDFS CLI on {SSH_USER}@{SSH_HOST}. Checked PATH and: {checked}. "
        "Make sure this SSH host is the Hadoop master, set HDFS_CONTAINER to the Hadoop container name, "
        "or set HDFS_CMD to the remote hdfs path."
    )


def list_remote_data_files(ssh):
    command = (
        f"if [ -d {q(REMOTE_DATA_DIR)} ]; then "
        f"find {q(REMOTE_DATA_DIR)} -type f \\( -name '*.csv' -o -name '*.json' \\); "
        "fi"
    )
    exit_code, stdout_text, _ = run_with_output(ssh, command)
    if exit_code != 0 or not stdout_text:
        return []
    return [line.strip() for line in stdout_text.splitlines() if line.strip()]


def upload_one(ssh, sftp, hdfs_cmd, hdfs_container, source_path, rel_path, source_is_remote):
    hdfs_path = posixpath.join(HDFS_BASE_DIR, rel_path.replace("\\", "/"))
    hdfs_dir = posixpath.dirname(hdfs_path)

    if source_is_remote:
        host_tmp_path = source_path
        remove_host_tmp = False
    else:
        host_tmp_path = posixpath.join(REMOTE_TMP, posixpath.basename(rel_path))
        sftp.put(source_path, host_tmp_path)
        remove_host_tmp = True

    hdfs_tmp_path = host_tmp_path

    if hdfs_container:
        hdfs_tmp_path = posixpath.join(REMOTE_TMP, posixpath.basename(rel_path))
        run(
            ssh,
            docker_cp_to_container(hdfs_container, host_tmp_path, hdfs_tmp_path)
        )

    mkdir_cmd = f"{q(hdfs_cmd)} dfs -mkdir -p {q(hdfs_dir)}"
    if hdfs_container:
        mkdir_cmd = docker_exec(hdfs_container, mkdir_cmd)
    run(ssh, mkdir_cmd)

    put_cmd = (
        f"{q(hdfs_cmd)} dfs -put -f "
        f"{q(hdfs_tmp_path)} "
        f"{q(hdfs_path)}"
    )
    if hdfs_container:
        put_cmd = docker_exec(hdfs_container, put_cmd)
    success = run(ssh, put_cmd)

    if success:
        print(f"✅ {hdfs_path}")

    if hdfs_container:
        run(
            ssh,
            docker_exec(hdfs_container, f"rm -f {q(hdfs_tmp_path)}")
        )
    if remove_host_tmp:
        run(ssh, f"rm -f {q(host_tmp_path)}")


def main():

    ssh = paramiko.SSHClient()

    ssh.set_missing_host_key_policy(
        paramiko.AutoAddPolicy()
    )

    connect_kwargs = {
        "hostname": SSH_HOST,
        "username": SSH_USER,
    }
    if SSH_KEY_PATH:
        connect_kwargs["key_filename"] = SSH_KEY_PATH
    elif SSH_PASSWORD:
        connect_kwargs["password"] = SSH_PASSWORD

    ssh.connect(**connect_kwargs)

    sftp = ssh.open_sftp()

    hdfs_target = resolve_hdfs_target(ssh)
    hdfs_cmd = hdfs_target["cmd"]
    hdfs_container = hdfs_target["container"]
    if hdfs_container:
        print(f"Using HDFS CLI: {hdfs_cmd} inside container {hdfs_container}")
    else:
        print(f"Using HDFS CLI: {hdfs_cmd}")

    remote_files = list_remote_data_files(ssh)
    if remote_files:
        print(f"Uploading from remote data dir: {REMOTE_DATA_DIR}")
        for remote_path in remote_files:
            rel_path = posixpath.relpath(remote_path, REMOTE_DATA_DIR)
            upload_one(
                ssh,
                sftp,
                hdfs_cmd,
                hdfs_container,
                remote_path,
                rel_path,
                source_is_remote=True,
            )
    else:
        print(f"Remote data dir is empty or missing, uploading from local: {DATA_DIR}")
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if not file.endswith((".csv", ".json")):
                    continue

                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, DATA_DIR).replace("\\", "/")
                upload_one(
                    ssh,
                    sftp,
                    hdfs_cmd,
                    hdfs_container,
                    local_path,
                    rel_path,
                    source_is_remote=False,
                )

    sftp.close()
    ssh.close()

    print("\n🎉 Upload completed")


if __name__ == "__main__":
    main()
