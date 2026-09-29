import { Component, type ReactNode } from 'react';
import type { BodyParams } from '@dt/avatar-core';
import { Avatar } from '../avatar/Avatar';
import { useAvatarLoadStore } from '../../store/avatarLoadStore';
import { PlaceholderAvatar } from './PlaceholderAvatar';

/** Falls back to the placeholder figure when the real avatar cannot load; the message is shown by the Viewer overlay. */
class AvatarBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  override state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  override componentDidCatch(error: Error): void {
    useAvatarLoadStore.getState().setError(error.message);
  }

  override render(): ReactNode {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

export function AvatarSlot({ params, poseId }: { params: BodyParams; poseId: string }) {
  return (
    <AvatarBoundary fallback={<PlaceholderAvatar params={params} poseId={poseId} />}>
      <Avatar />
    </AvatarBoundary>
  );
}
