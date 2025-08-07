from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    from .routes.stories import bp as stories_bp
    app.register_blueprint(stories_bp)

    return app

