import { expect, test } from '@playwright/test';

test('page loads with canvas and title', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('canvas')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Dijital İkiz' })).toBeVisible();
});
