import { createRoot } from 'react-dom/client';
import '../../../shared/i18n';
import '../../../index.css';
import { FacePanel, useFaceStore } from '../index';

// Test hook: lets e2e read the store (status, revision, overlay size) without any app route.
(window as unknown as { __faceStore: typeof useFaceStore }).__faceStore = useFaceStore;

const root = document.getElementById('root');
if (!root) throw new Error('Root element not found');
createRoot(root).render(
  <div className="mx-auto max-w-md p-4">
    <FacePanel />
  </div>,
);
