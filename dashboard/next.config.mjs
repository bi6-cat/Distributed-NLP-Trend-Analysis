/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Don't attempt to resolve these server-side only node modules on the client
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
        os: false,
        crypto: false,
        stream: false,
      };
    }
    return config;
  },
};

export default nextConfig;
