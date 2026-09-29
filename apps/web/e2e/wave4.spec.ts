import { expect, test, type Page, type Request } from '@playwright/test';

test.use({ channel: 'chrome' });

async function fill(page: Page, testId: string, value: string): Promise<void> {
  await page.getByTestId(testId).fill(value);
}

/** Fills one chart row (S/M/L) of the open add-item form. */
async function fillRow(page: Page, measure: string, values: string[]): Promise<void> {
  for (const [index, value] of values.entries()) await fill(page, `cell-${measure}-${index}`, value);
}

test('wardrobe: t-shirt with a size chart, fit report, heatmap, pants + shoes, take off', async ({ page }) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  const mutating: Request[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.location().url.endsWith('/favicon.ico')) errors.push(message.text());
  });
  page.on('request', (request) => {
    if (request.postData() || ['POST', 'PUT', 'PATCH'].includes(request.method())) mutating.push(request);
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });

  // --- open the wardrobe; the catalogue lists the templates with licence badges
  await page.getByRole('tab', { name: 'Gardırop', exact: true }).click();
  await expect(page.getByTestId('wardrobe-panel')).toBeVisible();
  await expect(page.getByTestId('template-tshirt')).toBeVisible();
  await expect(page.getByTestId('license-tshirt')).toHaveText('CC0');
  await expect(page.getByTestId('license-jeans')).toHaveText('CC BY');
  await expect(page.getByTestId('worn-empty')).toBeVisible();

  // --- add a t-shirt store item: validation first, then a valid S/M/L chart
  await page.getByTestId('use-template-tshirt').click();
  await page.getByTestId('form-save').click();
  await expect(page.getByTestId('form-errors')).toContainText('Ürüne bir ad verin');
  await expect(page.getByTestId('form-errors')).toContainText('Göğüs');
  await fill(page, 'form-name', 'Basic pamuklu tişört');
  await fill(page, 'form-url', 'not a url');
  await page.getByTestId('form-save').click();
  await expect(page.getByTestId('form-errors')).toContainText('geçerli bir adres');
  await fill(page, 'form-url', 'https://magaza.example/urun/123');

  // colour from a local "product photo"
  await page.evaluate(async () => {
    // a synthetic product photo: a red garment on a white background, handed to the file input like a user pick
    const canvas = document.createElement('canvas');
    canvas.width = 60;
    canvas.height = 60;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, 60, 60);
    context.fillStyle = '#d02020';
    context.fillRect(12, 12, 36, 36);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
    const transfer = new DataTransfer();
    transfer.items.add(new File([blob!], 'urun.png', { type: 'image/png' }));
    const input = document.querySelector<HTMLInputElement>('[data-testid="form-color-file"]')!;
    input.files = transfer.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await expect(page.getByTestId('form-color-hex')).toHaveValue(/^#[cd][0-9a-f][1-3][0-9a-f][1-3][0-9a-f]$/i, { timeout: 10_000 });
  await fill(page, 'form-color-hex', '#3b6ea8'); // a calmer blue for the screenshots

  await fillRow(page, 'chest', ['98', '106', '114']);
  await fillRow(page, 'waist', ['90', '98', '106']);
  await fillRow(page, 'length', ['68', '70', '72']);
  await fillRow(page, 'sleeve', ['19', '20', '21']);
  await page.getByTestId('form-selected-size').selectOption('M');
  // the filled form on a tall viewport, for the docs
  await page.setViewportSize({ width: 1440, height: 2100 });
  await page.waitForTimeout(400);
  await page.locator('aside').screenshot({ path: '../../docs/screenshots/wave4-form.png' });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByTestId('form-save-wear').click();

  // --- worn: fit chips, overall verdict and a recommended size
  await expect(page.getByTestId('worn-top')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('fit-overall')).toBeVisible();
  await expect(page.getByTestId('fit-region-chest')).toBeVisible();
  await expect(page.getByTestId('fit-region-waist')).toBeVisible();
  await expect(page.getByTestId('fit-region-sleeve')).toHaveAttribute('data-verdict', 'info'); // short sleeve: information only
  await expect(page.getByTestId('fit-recommended')).toContainText('Önerilen beden: M');
  const overallM = await page.getByTestId('fit-overall').getAttribute('data-verdict');
  const chestM = await page.getByTestId('fit-region-chest').textContent();

  // the garment mounted on the avatar: screenshot after the garment is fitted
  await page.getByRole('button', { name: 'Tam vücut', exact: true }).first().click();
  await page.waitForTimeout(4000);
  await page.screenshot({ path: '../../docs/screenshots/wave4-tshirt.png' });

  // --- heatmap toggle with legend
  await page.getByTestId('heatmap-toggle').check();
  await expect(page.getByTestId('heatmap-legend')).toBeVisible();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: '../../docs/screenshots/wave4-heatmap.png' });
  await page.getByTestId('heatmap-toggle').uncheck();
  await expect(page.getByTestId('heatmap-legend')).toHaveCount(0);

  // --- switching the size changes the report
  await page.getByTestId('size-top-S').click();
  await expect(page.getByTestId('fit-report-top')).toHaveAttribute('data-size', 'S');
  await expect(page.getByTestId('fit-overall')).toHaveAttribute('data-verdict', 'tight');
  expect(await page.getByTestId('fit-region-chest').textContent()).not.toBe(chestM);
  await page.getByTestId('size-top-L').click();
  await expect(page.getByTestId('fit-report-top')).toHaveAttribute('data-size', 'L');
  await expect(page.getByTestId('fit-overall')).not.toHaveAttribute('data-verdict', 'tight');
  await page.getByTestId('fit-use-recommended').click(); // "Bu bedeni seç" jumps back to the recommended M
  await expect(page.getByTestId('fit-report-top')).toHaveAttribute('data-size', 'M');
  await expect(page.getByTestId('fit-overall')).toHaveAttribute('data-verdict', overallM ?? '');

  // the body slider moves the report too: a much slimmer chest makes the M shirt looser
  await page.getByRole('tab', { name: 'Ölçüler', exact: true }).click();
  const chest = page.getByRole('spinbutton', { name: 'Göğüs çevresi', exact: true });
  await chest.fill('84');
  await chest.blur();
  await page.getByRole('tab', { name: 'Gardırop', exact: true }).click();
  await expect(page.getByTestId('fit-region-chest')).toHaveAttribute('data-verdict', /loose|oversized/, { timeout: 20_000 });
  await page.getByRole('tab', { name: 'Ölçüler', exact: true }).click();
  await chest.fill('100');
  await chest.blur();
  await page.getByRole('tab', { name: 'Gardırop', exact: true }).click();
  await expect(page.getByTestId('fit-region-chest')).toHaveAttribute('data-verdict', 'regular', { timeout: 20_000 });

  // --- pants (chart in flat measurements, doubled on save)
  await page.getByTestId('wardrobe-add').click();
  await page.getByTestId('form-template').selectOption('jeans');
  await fill(page, 'form-name', 'Slim jean');
  await page.getByTestId('form-flat').check();
  await fillRow(page, 'waist', ['40', '43', '46']);
  await fillRow(page, 'hip', ['49', '52', '55']);
  await fillRow(page, 'inseam', ['78', '79', '80']);
  await fillRow(page, 'thigh', ['30', '31.5', '33']);
  await page.getByTestId('form-save-wear').click();
  await expect(page.getByTestId('worn-bottom')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('fit-report-bottom').getByTestId('fit-region-waist')).toContainText('cm');
  await expect(page.getByTestId('fit-report-bottom').getByTestId('fit-recommended')).toBeVisible();

  // --- shoes from EU size labels
  await page.getByTestId('wardrobe-add').click();
  await page.getByTestId('form-template').selectOption('sneakers');
  await fill(page, 'form-name', 'Koşu ayakkabısı');
  await page.getByTestId('form-selected-size').selectOption('42');
  await page.getByTestId('form-save-wear').click();
  await expect(page.getByTestId('worn-shoes')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('shoes-hint')).toBeVisible();
  await expect(page.getByTestId('fit-report-shoes').getByTestId('fit-region-footLength')).toBeVisible();

  await page.getByRole('button', { name: 'Tam vücut', exact: true }).first().click();
  await page.waitForTimeout(3500);
  await page.screenshot({ path: '../../docs/screenshots/wave4-outfit.png' });
  await page.getByTestId('focus-feet').click();
  await page.waitForTimeout(3500);
  await page.screenshot({ path: '../../docs/screenshots/wave4-feet-shoes.png' });

  // --- items survive a reload (IndexedDB) and the outfit is restored
  await page.reload();
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });
  await page.getByRole('tab', { name: 'Gardırop', exact: true }).click();
  await expect(page.getByTestId('worn-top')).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId('worn-bottom')).toBeVisible();
  await expect(page.getByTestId('worn-shoes')).toBeVisible();
  await expect(page.getByTestId('item-list').locator('li')).toHaveCount(3);

  // --- take everything off
  await page.getByTestId('take-off-shoes').click();
  await page.getByTestId('take-off-bottom').click();
  await page.getByTestId('take-off-top').click();
  await expect(page.getByTestId('worn-empty')).toBeVisible();
  await page.getByRole('button', { name: 'Tam vücut', exact: true }).first().click();
  await page.waitForTimeout(2500);
  await page.screenshot({ path: '../../docs/screenshots/wave4-takeoff.png' });

  // nothing was uploaded
  expect(mutating.map((r) => `${r.method()} ${r.url()}`)).toEqual([]);
  expect(errors).toEqual([]);
});
