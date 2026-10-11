"""Optional *native* end-to-end browser smoke test for a locally running API.

Run from the project root, after starting the API:
    python -m pip install playwright
    python -m playwright install chromium
    python tests/browser_smoke.py

Unlike a mocked browser, this navigates the real URL and uses real network
requests, and therefore should be run on your own machine for final sign-off.
"""
import argparse
import csv
import io
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000/')
    parser.add_argument('--show-browser', action='store_true')
    args = parser.parse_args()
    checked = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.show_browser)
        page = browser.new_page(accept_downloads=True)
        dialogs = []
        page.on('dialog', lambda dialog: (dialogs.append(dialog.message), dialog.accept()))
        errors = []
        page.on('pageerror', lambda err: errors.append(str(err)))
        page.goto(args.url, wait_until='domcontentloaded', timeout=45000)
        assert page.title().startswith('Global PropPredict')
        page.wait_for_timeout(300)
        assert page.locator('#api-banner').is_hidden(), 'API health check failed'
        checked.append('Dashboard and API health')

        page.locator('#listing_price').fill('10000000')
        page.locator('#age_years').fill('19')
        page.locator('#rera_approved').select_option('0')
        page.locator('button[onclick="nextStep(2)"]').click()
        page.locator('#city_state').fill('Chennai')
        page.locator('#dist_bus').fill('6.7')
        payload = page.evaluate('buildPayload(10000000)')
        assert payload['age_years'] == 19 and payload['dist_bus'] == 6.7
        page.locator('#run-model-btn').click()
        page.wait_for_function('!document.querySelector("#results").classList.contains("hidden")')
        page.wait_for_timeout(1700)
        assert '₹' in page.locator('#final-price').inner_text()
        checked.append('India prediction with editable features')

        page.locator('#btn-investment').click()
        assert page.locator('#inv-score-val').inner_text() == '—'
        for key, value in [('inv_cagr', '7'), ('inv_yield', '4'), ('inv_infra', '7')]:
            page.locator('#' + key).fill(value)
        page.locator('#btn-investment').click()
        page.wait_for_function('document.querySelector("#inv-score-val").innerText.includes("/ 100")')
        score1 = page.locator('#inv-score-val').inner_text()
        page.locator('#inv_cagr').fill('14')
        page.locator('#btn-investment').click()
        page.wait_for_timeout(300)
        assert score1 != page.locator('#inv-score-val').inner_text()
        checked.append('User-controlled investment scenario')

        assert 'Illustrative' in page.locator('#future-sub').inner_text()
        page.get_by_role('button', name='Calculate Breakeven').click()
        page.wait_for_function('document.querySelector("#rent-buy-verdict").innerText.length > 10')
        checked.append('Future projection and rent-vs-buy')

        page.get_by_role('button', name='New Prediction').click()
        page.locator('#btn-us').click()
        page.locator('#year_built').fill('2010')
        page.locator('#year_remodeled').fill('2021')
        page.locator('#listing_price').fill('600000')
        page.locator('button[onclick="nextStep(2)"]').click()
        page.locator('#transit_score').fill('80')
        page.locator('#fire_zone').select_option('1')
        assert page.evaluate('buildPayload(600000).transit_score') == 80
        page.locator('#run-model-btn').click()
        page.wait_for_function('!document.querySelector("#results").classList.contains("hidden")')
        page.wait_for_timeout(1700)
        assert '$' in page.locator('#final-price').inner_text()
        checked.append('USA prediction with editable features')

        row = dict(area=1200, bedrooms=2, bathrooms=2, floor=3, total_floors=10,
                   dist_metro=2, dist_bus=2, dist_railway=5, dist_highway=3,
                   dist_it_park=5, dist_school=1, age_years=5, rera_approved=1,
                   city='Chennai', facing='East', builder='DLF')
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
        page.locator('#csv-upload').set_input_files({
            'name': 'example.csv', 'mimeType': 'text/csv', 'buffer': output.getvalue().encode()
        })
        page.locator('#btn-bulk').click()
        page.wait_for_function('!document.querySelector("#bulk-results").classList.contains("hidden")')
        assert page.locator('#bulk-count').inner_text() == '1'
        with page.expect_download() as download:
            page.get_by_role('button', name='Download CSV').click()
        assert download.value.suggested_filename == 'BulkResults.csv'
        checked.append('Bulk CSV upload, predictions and download')

        page.locator('#btn-compare').click()
        page.wait_for_function('!document.querySelector("#compare-results").classList.contains("hidden")')
        page.locator('#cmp-b-us').click()
        page.locator('#btn-compare').click()
        page.wait_for_function('document.querySelector("#cmp-winner").innerText === "Different currencies"')
        checked.append('Property comparison and cross-currency handling')

        page.get_by_role('button', name='New Prediction').click()
        page.locator('#btn-in').click()
        page.locator('#floor').fill('20')
        page.locator('#total_floors').fill('5')
        page.locator('button[onclick="nextStep(2)"]').click()
        page.locator('#run-model-btn').click()
        page.wait_for_timeout(700)
        assert dialogs and 'Property floor cannot exceed total floors' in dialogs[-1]
        assert page.locator('#results').is_hidden()
        checked.append('Bad input produces useful error, not fake prediction')
        assert not errors, f'Uncaught page errors: {errors}'
        checked.append('No uncaught JavaScript errors')
        browser.close()
    for item in checked:
        print('PASS:', item)
    print(f'{len(checked)} browser checks passed.')


if __name__ == '__main__':
    main()
