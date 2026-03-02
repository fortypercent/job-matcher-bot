import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests

load_dotenv()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет!\n\nНапиши специальность:\nPython Developer\nData Scientist\nJavaScript Developer"
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    spec = update.message.text
    await update.message.reply_text(f"⏳ Ищу '{spec}'...")
    
    try:
        response = requests.get("https://api.hh.ru/vacancies", params={
            'text': spec, 'area': 1, 'per_page': 50
        }, timeout=10)
        
        jobs = response.json().get('items', [])[:5]
        
        if not jobs:
            await update.message.reply_text("😔 Вакансий не найдено")
            return
        
        for i, job in enumerate(jobs, 1):
            salary_text = ""
            if job.get('salary'):
                salary_text = f"\n💰 {job['salary'].get('from', '?')}-{job['salary'].get('to', '?')} RUB"
            
            msg = f"{i}. {job['name']}\n🏢 {job['employer']['name']}{salary_text}\n🔗 {job['alternate_url']}"
            await update.message.reply_text(msg, disable_web_page_preview=True)
    
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")

def main():
    token = os.getenv('BOT_TOKEN')
    
    if not token:
        print("❌ BOT_TOKEN не найден в .env файле!")
        return
    
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("🤖 Бот запущен! Нажми Ctrl+C чтобы остановить")
    app.run_polling()

if __name__ == '__main__':
    main()