module.exports = {
  apps: [
    {
      name: "dashboard",
      cwd: "/root/Bigdata/Distributed-NLP-Trend-Analysis/dashboard",
      script: "npm",
      args: "run start -- --hostname 0.0.0.0 --port 3000",
      env: {
        NODE_ENV: "production",
      },
    },
  ],
};
