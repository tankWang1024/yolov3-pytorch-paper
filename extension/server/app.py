from flask import Flask
from flask import render_template
from flask import url_for
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')

# 环境变量配置
# set FLASK_APP=server.py
# $env:FLASK_APP
@app.route('/')
def helloworld():
    return render_template('index.html', data="static/1.jpg", name="现在刚刚开始做后端应用")

if __name__ == '__main__':
    app.run()