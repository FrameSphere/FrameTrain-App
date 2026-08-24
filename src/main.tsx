import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import { installGlobalErrorReporting } from './utils/errorReport';
import { AppErrorBoundary } from './components/AppErrorBoundary';

// Globales Auto-Error-Reporting an den Manager (speist die Auto-Fix-Pipeline).
installGlobalErrorReporting();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </React.StrictMode>,
);
