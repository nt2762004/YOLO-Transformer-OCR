const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  page.setDefaultTimeout(90000);

  try {
    console.log('--- Đang truy cập App ---');
    // Thêm tham số ngẫu nhiên để tránh cache của Streamlit
    await page.goto('https://yolo-transformer-ocr.streamlit.app/?t=' + Date.now(), { 
      waitUntil: 'domcontentloaded' 
    });

    // Chờ xem có nút Wake up không (đợi 15s)
    const wakeUpButton = await page.waitForSelector('text=Yes, get this app back up!', { timeout: 15000 }).catch(() => null);

    if (wakeUpButton) {
      console.log('--- Phát hiện app đang ngủ. Đang nhấn Wake Up... ---');
      await wakeUpButton.click({ force: true });
      
      console.log('--- Đã nhấn! Chờ giao diện chính xuất hiện (max 60s)... ---');
      // Chờ selector đặc trưng của Streamlit để chắc chắn nó đã load xong
      await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 60000 });
      console.log('--- App đã thức giấc thành công! ---');
    } else {
      console.log('--- Không thấy nút Wake Up. Kiểm tra giao diện chính... ---');
      const isLoaded = await page.$('[data-testid="stAppViewContainer"]');
      if (isLoaded) {
        console.log('--- App vẫn đang thức, không cần làm gì thêm. ---');
      } else {
        console.log('--- Không thấy nút nhưng cũng không thấy giao diện. Có thể app đang load hoặc lỗi. ---');
      }
    }
  } catch (err) {
    console.log('Có lỗi hoặc timeout: ' + err.message);
  } finally {
    await browser.close();
    console.log('--- Kết thúc session ---');
  }
})();
