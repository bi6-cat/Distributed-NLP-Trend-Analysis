# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/jammy64"
  config.vm.box_check_update = false

  # Shell script để tự động tạo user 'zett' trên tất cả các máy ảo
  # Giúp đồng bộ hoàn toàn với cấu hình Ansible hiện tại của dự án
  $setup_user_script = <<-SCRIPT
    if ! id -u zett > /dev/null 2>&1; then
      useradd -m -s /bin/bash zett
      echo "zett:0008" | chpasswd
      echo "zett ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/zett
      
      # Copy SSH keys từ user vagrant sang user zett để Ansible kết nối được dễ dàng
      mkdir -p /home/zett/.ssh
      cp /home/vagrant/.ssh/authorized_keys /home/zett/.ssh/
      chown -R zett:zett /home/zett/.ssh
      chmod 700 /home/zett/.ssh
      chmod 600 /home/zett/.ssh/authorized_keys
    fi
  SCRIPT

  config.vm.provision "shell", inline: $setup_user_script

  # 1. Master Node (Brain, NameNode, Spark Master, Airflow)
  config.vm.define "master" do |master|
    master.vm.hostname = "master"
    master.vm.network "private_network", ip: "192.168.56.11"
    master.vm.provider "virtualbox" do |vb|
      vb.name = "nlp-master"
      vb.memory = "4096"  # Reduced from 8GB to fit 16GB host RAM
      vb.cpus = 2
    end
  end

  # 2. Worker Node 1 (DataNode, Spark Worker)
  config.vm.define "worker1" do |worker1|
    worker1.vm.hostname = "worker1"
    worker1.vm.network "private_network", ip: "192.168.56.12"
    worker1.vm.provider "virtualbox" do |vb|
      vb.name = "nlp-worker-1"
      vb.memory = "5096"
      vb.cpus = 6
    end
  end

  # 3. Worker Node 2 (DataNode, Spark Worker)
  config.vm.define "worker2" do |worker2|
    worker2.vm.hostname = "worker2"
    worker2.vm.network "private_network", ip: "192.168.56.13"
    worker2.vm.provider "virtualbox" do |vb|
      vb.name = "nlp-worker-2"
      vb.memory = "5096"
      vb.cpus = 6
    end
  end

  # 4. Storage Node (ClickHouse)
  config.vm.define "storage" do |storage|
    storage.vm.hostname = "storage"
    storage.vm.network "private_network", ip: "192.168.56.14"
    storage.vm.provider "virtualbox" do |vb|
      vb.name = "nlp-storage"
      vb.memory = "4096"
      vb.cpus = 2
    end
  end

end
