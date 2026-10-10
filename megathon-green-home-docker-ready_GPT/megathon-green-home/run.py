from app.main import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(__import__('os').getenv('PORT','8000')), debug=False)
