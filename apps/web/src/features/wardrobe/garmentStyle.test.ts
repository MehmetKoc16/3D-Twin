import { describe, expect, it } from 'vitest';
import { garmentStyle, isTuckedTop } from './garmentStyle';

describe('garmentStyle', () => {
  it('marks the tucked t-shirt template as tucked and the sweater as untucked', () => {
    expect(garmentStyle({ id: 'tshirt', category: 'top' })).toBe('tucked');
    expect(isTuckedTop({ id: 'tshirt', category: 'top' })).toBe(true);
    expect(garmentStyle({ id: 'sweatshirt', category: 'top' })).toBe('untucked');
  });

  it('defaults to untucked and never tucks bottoms or shoes', () => {
    expect(garmentStyle({ id: 'some-new-top', category: 'top' })).toBe('untucked');
    expect(garmentStyle({ id: 'tshirt', category: 'bottom' })).toBe('untucked');
    expect(isTuckedTop({ id: 'jeans', category: 'bottom' })).toBe(false);
  });
});
