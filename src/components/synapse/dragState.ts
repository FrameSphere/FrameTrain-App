// Shared drag state — module-level object shared across the component tree.
// This bypasses dataTransfer MIME-type issues in Tauri's WKWebView entirely.
export const dragState = {
  nodeType: null as string | null,
};
