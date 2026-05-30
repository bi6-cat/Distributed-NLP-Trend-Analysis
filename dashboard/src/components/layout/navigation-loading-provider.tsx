"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { usePathname } from "next/navigation";

type NavigationLoadingContextValue = {
  startLoading: () => void;
};

const NavigationLoadingContext = createContext<NavigationLoadingContextValue | null>(null);

export function NavigationLoadingProvider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [isLoading, setIsLoading] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const previousPathnameRef = useRef(pathname);

  const clearLoadingTimer = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const startLoading = useCallback(() => {
    clearLoadingTimer();
    setIsLoading(true);
    timeoutRef.current = setTimeout(() => setIsLoading(false), 8000);
  }, [clearLoadingTimer]);

  useEffect(() => {
    if (pathname === previousPathnameRef.current) return;

    previousPathnameRef.current = pathname;
    if (!isLoading) return;

    clearLoadingTimer();
    timeoutRef.current = setTimeout(() => setIsLoading(false), 250);

    return clearLoadingTimer;
  }, [pathname, isLoading, clearLoadingTimer]);

  const value = useMemo(() => ({ startLoading }), [startLoading]);

  return (
    <NavigationLoadingContext.Provider value={value}>
      {isLoading && (
        <div className="fixed inset-x-0 top-0 z-50 h-1 overflow-hidden bg-indigo-100">
          <div className="h-full w-1/2 animate-[dashboard-progress_1.1s_ease-in-out_infinite] rounded-r-full bg-gradient-to-r from-indigo-500 via-violet-500 to-emerald-400" />
        </div>
      )}
      {children}
    </NavigationLoadingContext.Provider>
  );
}

export function useNavigationLoading() {
  const context = useContext(NavigationLoadingContext);

  if (!context) {
    return { startLoading: () => undefined };
  }

  return context;
}
