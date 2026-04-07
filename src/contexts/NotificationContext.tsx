import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';
import { useTheme } from './ThemeContext';

/**
 * FrameTrain Notification System
 * 
 * Universelles Benachrichtigungssystem für die Desktop-App.
 * 
 * VERWENDUNG:
 * 
 * 1. Importiere den Hook:
 *    import { useNotification } from '../contexts/NotificationContext';
 * 
 * 2. Im Component:
 *    const { notify, success, error, warning, info } = useNotification();
 * 
 * 3. Notification anzeigen:
 *    success('Erfolgreich gespeichert!');
 *    error('Fehler beim Laden');
 *    warning('Achtung: Aktion wird überschrieben');
 *    info('Neue Version verfügbar');
 * 
 *    // Oder mit mehr Optionen:
 *    notify({
 *      type: 'success',
 *      title: 'Training gestartet',
 *      message: 'Das Modell wird jetzt trainiert...',
 *      duration: 5000, // ms, 0 = permanent
 *      action: {
 *        label: 'Anzeigen',
 *        onClick: () => navigateToTraining()
 *      }
 *    });
 */

// ============ Types ============

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface NotificationAction {
  label: string;
  onClick: () => void;
}

export interface NotificationOptions {
  type: NotificationType;
  title?: string;
  message: string;
  duration?: number; // ms, default 4000, 0 = permanent
  action?: NotificationAction;
}

interface Notification extends NotificationOptions {
  id: string;
}

interface NotificationContextType {
  notifications: Notification[];
  notify: (options: NotificationOptions) => string;
  success: (message: string, title?: string) => string;
  error: (message: string, title?: string) => string;
  warning: (message: string, title?: string) => string;
  info: (message: string, title?: string) => string;
  dismiss: (id: string) => void;
  dismissAll: () => void;
}

// ============ Context ============

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

// ============ Provider Component ============

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const dismiss = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  const dismissAll = useCallback(() => {
    setNotifications([]);
  }, []);

  const notify = useCallback((options: NotificationOptions): string => {
    const id = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const notification: Notification = {
      ...options,
      id,
      duration: options.duration ?? 4000,
    };

    setNotifications(prev => [...prev, notification]);

    // Auto-dismiss nach duration (wenn nicht 0)
    if (notification.duration > 0) {
      setTimeout(() => {
        dismiss(id);
      }, notification.duration);
    }

    return id;
  }, [dismiss]);

  // Shorthand-Funktionen für häufige Typen
  const success = useCallback((message: string, title?: string) => {
    return notify({ type: 'success', message, title });
  }, [notify]);

  const error = useCallback((message: string, title?: string) => {
    return notify({ type: 'error', message, title, duration: 6000 }); // Errors länger anzeigen
  }, [notify]);

  const warning = useCallback((message: string, title?: string) => {
    return notify({ type: 'warning', message, title, duration: 5000 });
  }, [notify]);

  const info = useCallback((message: string, title?: string) => {
    return notify({ type: 'info', message, title });
  }, [notify]);

  return (
    <NotificationContext.Provider
      value={{
        notifications,
        notify,
        success,
        error,
        warning,
        info,
        dismiss,
        dismissAll,
      }}
    >
      {children}
      <NotificationContainer notifications={notifications} onDismiss={dismiss} />
    </NotificationContext.Provider>
  );
}

// ============ Notification Container ============

interface NotificationContainerProps {
  notifications: Notification[];
  onDismiss: (id: string) => void;
}

function NotificationContainer({ notifications, onDismiss }: NotificationContainerProps) {
  return (
    <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-3 max-w-sm w-full pointer-events-none">
      {notifications.map(notification => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onDismiss={() => onDismiss(notification.id)}
        />
      ))}
    </div>
  );
}

// ============ Single Notification Item ============

interface NotificationItemProps {
  notification: Notification;
  onDismiss: () => void;
}

function NotificationItem({ notification, onDismiss }: NotificationItemProps) {
  const { currentTheme } = useTheme();

  // Icon und Farben basierend auf Typ
  const config = {
    success: {
      icon: CheckCircle,
      bgColor: 'bg-emerald-500/20',
      borderColor: 'border-emerald-500/50',
      iconColor: 'text-emerald-400',
      titleColor: 'text-emerald-300',
    },
    error: {
      icon: AlertCircle,
      bgColor: 'bg-red-500/20',
      borderColor: 'border-red-500/50',
      iconColor: 'text-red-400',
      titleColor: 'text-red-300',
    },
    warning: {
      icon: AlertTriangle,
      bgColor: 'bg-amber-500/20',
      borderColor: 'border-amber-500/50',
      iconColor: 'text-amber-400',
      titleColor: 'text-amber-300',
    },
    info: {
      icon: Info,
      bgColor: 'bg-blue-500/20',
      borderColor: 'border-blue-500/50',
      iconColor: 'text-blue-400',
      titleColor: 'text-blue-300',
    },
  };

  const typeConfig = config[notification.type];
  const Icon = typeConfig.icon;

  return (
    <div
      className={`
        pointer-events-auto
        ${typeConfig.bgColor} 
        ${typeConfig.borderColor}
        border
        backdrop-blur-xl
        rounded-xl
        shadow-2xl
        p-4
        animate-slide-in-right
        transition-all
        duration-300
        hover:scale-[1.02]
      `}
      role="alert"
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className={`flex-shrink-0 ${typeConfig.iconColor}`}>
          <Icon className="w-5 h-5" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {notification.title && (
            <p className={`font-semibold text-sm ${typeConfig.titleColor}`}>
              {notification.title}
            </p>
          )}
          <p className="text-white/90 text-sm mt-0.5">
            {notification.message}
          </p>

          {/* Action Button */}
          {notification.action && (
            <button
              onClick={() => {
                notification.action?.onClick();
                onDismiss();
              }}
              className={`
                mt-2
                text-sm
                font-medium
                ${typeConfig.iconColor}
                hover:underline
                transition-all
              `}
            >
              {notification.action.label}
            </button>
          )}
        </div>

        {/* Close Button */}
        <button
          onClick={onDismiss}
          className="flex-shrink-0 text-white/50 hover:text-white transition-colors"
          aria-label="Schließen"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

// ============ Hook ============

export function useNotification() {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
}
