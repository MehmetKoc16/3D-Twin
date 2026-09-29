import type { BodyParams } from '@dt/avatar-core';
import { PlaceholderAvatar } from './PlaceholderAvatar';

export function AvatarSlot({ params, poseId }: { params: BodyParams; poseId: string }) {
  return <PlaceholderAvatar params={params} poseId={poseId} />;
}

