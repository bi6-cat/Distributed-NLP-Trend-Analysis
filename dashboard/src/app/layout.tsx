import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";
import { NavigationLoadingProvider } from "@/components/layout/navigation-loading-provider";

export const metadata: Metadata = {
  title: "Tech Trend & Controversy Radar",
  description: "Vietnamese Tech Trend & Controversy Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased bg-slate-50 text-slate-900 selection:bg-indigo-100 selection:text-indigo-900">
        <NavigationLoadingProvider>
          <Sidebar />
          <div className="ml-64 flex flex-col min-h-screen">
            <Topbar />
            <main className="flex-1 px-8 lg:px-10 py-8">
              <div className="mx-auto max-w-[1400px]">
                {children}
              </div>
            </main>
          </div>
        </NavigationLoadingProvider>
      </body>
    </html>
  );
}
