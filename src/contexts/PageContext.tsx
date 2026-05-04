import { createContext, useContext, useState, ReactNode } from 'react';

/**
 * Globaler Context für den aktuellen Seiten-Inhalt
 * Damit der AI Coach automatisch den Kontext der aktuellen Seite erhält
 */

interface PageContextType {
  currentPageContent: string;
  setCurrentPageContent: (content: string) => void;
}

const PageContext = createContext<PageContextType | undefined>(undefined);

export function PageContextProvider({ children }: { children: ReactNode }) {
  const [currentPageContent, setCurrentPageContent] = useState('');

  return (
    <PageContext.Provider value={{ currentPageContent, setCurrentPageContent }}>
      {children}
    </PageContext.Provider>
  );
}

export function usePageContext() {
  const context = useContext(PageContext);
  if (!context) {
    throw new Error('usePageContext must be used within PageContextProvider');
  }
  return context;
}
