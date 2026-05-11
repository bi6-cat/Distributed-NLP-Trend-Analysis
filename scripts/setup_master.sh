#!/bin/bash
set -e

echo '>>> Updating package lists...'
sudo apt update -qq

echo '>>> Installing Ansible and sshpass...'
sudo DEBIAN_FRONTEND=noninteractive apt install ansible sshpass -y -qq

echo '>>> Copying Vagrant private keys to ~/.ssh/ (to fix 0777 permission issue on shared folder)...'
for NODE in master worker1 worker2 storage; do
    SRC="/vagrant/.vagrant/machines/${NODE}/virtualbox/private_key"
    DEST="$HOME/.ssh/id_rsa_${NODE}"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$DEST"
        chmod 600 "$DEST"
        echo "  -> Copied and secured key for ${NODE}"
    else
        echo "  -> WARNING: Key not found for ${NODE}: $SRC"
    fi
done

echo '>>> Done! Master is prepared.'
