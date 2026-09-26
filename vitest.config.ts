import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    globals: true,
    // 5 s Standard reichten auf einer ausgelasteten Maschine nicht: wechselnde
    // Tests liefen knapp darueber, ohne dass etwas falsch war. Das Limit ist
    // eine Obergrenze fuer Haenger, keine Messung.
    testTimeout: 15000,
  },
});
