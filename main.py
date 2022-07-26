import base64
import io
import logging

from PIL import Image
from collections import defaultdict
from difflib import SequenceMatcher
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
logger = app.logger
logger.setLevel(logging.DEBUG)
socketio = SocketIO(app)

game_players_map = defaultdict(list)  # {game_id: list of players' user_name}
game_sentences_map = defaultdict(list)  # {game_id: list of players' sentences}


@app.route('/')
def index():
    logger.info("hello!")
    return render_template('index.html')


@app.route('/welcome/<user_name>', methods=['GET'])
def welcome(user_name):
    logger.info(f"welcome {user_name}!")
    return render_template('welcome.html', user_name=user_name)


@app.route('/waiting_players/<game_id>/<user_name>', methods=['GET'])
def waiting_players(game_id, user_name):
    logger.info(f"game: {game_id}, {user_name} is waiting for other players!")
    return render_template('waiting_players.html', game_id=game_id, user_name=user_name)


@app.route('/game_loop/<game_id>/<user_name>', methods=['GET', 'POST'])
def game_loop(game_id, user_name):
    logger.info(f"game: {game_id}, user {user_name} enters game_loop")
    players = game_players_map[game_id]

    # TODO: random select a drawer_id once, then use socket.io to broadcast the decision to guessers
    # TODO: make the drawer selection and guesser order fair
    # drawer_id = randint(0, len(players) - 1)
    drawer_id = 0
    if len(players) > 0:
        drawer_name = players[drawer_id]
    else:
        logger.error("can't find a drawer, players are empty!")
    return render_template('game_loop.html', game_id=game_id, user_name=user_name, drawer_name=drawer_name)


def message_received(methods=['GET', 'POST']):
    logger.info('message was received!!!')


@socketio.on('join-game-event')
def handle_join_game_event(json, methods=['GET', 'POST']):
    logger.info('received join event: ' + str(json))
    game_id = json["game_id"]
    user_name = json["user_name"]
    if not game_id:
        logger.error(f"bad game id: {game_id}")
        return
    if not user_name:
        logger.error(f"bad user_name: {user_name}")
        return

    if user_name not in game_players_map[game_id]:
        game_players_map[game_id].append(user_name)
    else:
        logger.error(f"Duplicated user_name: {user_name} !")
    logger.info(f"return players: {game_players_map[game_id]}")
    socketio.emit('join-game-event-response', {"players": game_players_map[game_id]}, callback=message_received)


@socketio.on('drawer-submit-event')
def handle_drawer_submit_event(json, methods=['GET', 'POST']):
    logger.info('received drawer-submit-event: ' + str(json))
    game_sentences_map[json["game_id"]].clear()
    game_sentences_map[json["game_id"]].append(json["sentence"])
    socketio.emit('drawer-submit-event-response', json, callback=message_received)

    # TODO: call text-to-image model and save output image to ai_drawed_image_path
    # TODO: cache (draw's sentence, output image) for faster demo
    # Mock the output image from text-to-image model

    ai_drawed_image_path = "resources/pytorch_logo.png"

    ######################################################################

    img = get_encoded_img(ai_drawed_image_path)
    socketio.emit('ai-returns-image-event', img, callback=message_received)


@socketio.on('guesser-submit-event')
def handle_guesser_submit_event(json, methods=['GET', 'POST']):
    logger.info('received guesser-submit-event: ' + str(json))
    correct_sentence = game_sentences_map[json["game_id"]][0]
    guess_sentence = json["sentence"]

    # TODO: call text-similarity model to calculate the score

    similarity_score = SequenceMatcher(None, correct_sentence, guess_sentence).ratio() * 10

    ######################################################################

    json["score"] = similarity_score
    socketio.emit('guesser-submit-event-response', json, callback=message_received)


def get_encoded_img(image_path):
    img = Image.open(image_path, mode='r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return encoded_img


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
