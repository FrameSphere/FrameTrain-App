import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import type { PageId } from '../ai/coachContext';
import { setCurrentScreen } from '../utils/errorReport';

/**
 * Globaler Context für die aktuelle Seite des AI Coaches.
 * - currentPageContent: Live-Zustand der Seite (aus buildPageContext o.ä.)
 * - currentPageId:      welche Seite gerade aktiv ist → steuert das lazy
 *                       geladene, seiten-spezifische Wissen des Coaches.
 */

interface PageContextType {
  currentPageContent: string;
  currentPageId: PageId | null;
  /** Setzt Live-Zustand (+ optional die Seiten-ID für das seiten-spezifische Wissen). */
  setCurrentPageContent: (content: string, pageId?: PageId) => void;
}

const PageContext = createContext<PageContextType | undefined>(undefined);

export function PageContextProvider({ children }: { children: ReactNode }) {
  const [currentPageContent, setContent] = useState('');
  const [currentPageId, setPageId] = useState<PageId | null>(null);

  const setCurrentPageContent = useCallback((content: string, pageId?: PageId) => {
    setContent(content);
    if (pageId !== undefined) {
      setPageId(pageId);
      // Screen-Kontext für automatische Error-Reports mitführen.
      setCurrentScreen(pageId);
    }
  }, []);

  return (
    <PageContext.Provider value={{ currentPageContent, currentPageId, setCurrentPageContent }}>
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
