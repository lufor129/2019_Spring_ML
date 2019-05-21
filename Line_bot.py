from flask import Flask, request, abort
import tensorflow as tf
import cv2
import os
import numpy as np
import time
import json
import csv
import random

with open("CompareChart.json","r") as JsonReader:
    chart = json.loads(JsonReader.read())

with open("outputFinal.csv",newline='',encoding="utf8") as csvfile:
    rowtemp = csv.reader(csvfile, delimiter=',')
    rows = [row for row in rowtemp]

def getKey(label):
    for key,value in chart.items():
        if(label == value):
            return key
    return "0_"

def getImage(label):
    for row in rows:
        if(row[0] == label):
            return row

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

app = Flask(__name__)

line_bot_api = LineBotApi('qrlGKxcOxlOLQQVVaxi3qYNYgHZ/hyS5OtiQEhsWD4KSJoLmHgdZdGJNQCYXl5zdS+CO0SEIq809s2GywKDLuRdj6yBm3txaLhohYHAAYfKFS0afurL20790fftqwukeaEn9iX3zpNS++K/AD2aVfQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('4dd6dd02ea744078f03b2d5f9454cf64')
static_tmp_path = os.path.join(os.path.dirname(__file__), 'pictmp')

save_model_path = './model'
sess = tf.Session()
saver = tf.train.import_meta_graph(save_model_path+"/TransferModel.meta")
saver.restore(sess,tf.train.latest_checkpoint(save_model_path))
prediction = tf.get_collection('pred_network')[0]
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("input_x:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
logits  = graph.get_tensor_by_name("logits:0")
prediction = tf.argmax(logits,1)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=(ImageMessage, TextMessage))
def handle_message(event):
    if (event.message.type == "image"):
        picName = "./pictmp/"+str(int(time.time()))+".png"
        line_bot_api.push_message(event.source.user_id,TextSendMessage(text="讓我思考一下"))
        message_content = line_bot_api.get_message_content(event.message.id)
        with open(picName,"wb") as fd:
            for chunk in message_content.iter_content():
                fd.write(chunk)

        img = cv2.imdecode(np.fromfile(picName,dtype=np.uint8),-1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(224,224))

        label = sess.run(prediction,feed_dict={x:[img],keep_prob:1})
        label = getKey(label[0])

        Image = getImage(label)
        image_message = ImageSendMessage(
            original_content_url=Image[6],
            preview_image_url=Image[6]
        )
        line_bot_api.reply_message(event.reply_token,[image_message,TextSendMessage(text="畫作名稱:\n"+Image[1]),TextSendMessage(text=Image[3]),TextSendMessage(text="連結: \n"+Image[2])])
        return 0
    else:
        if(event.message.text == "隨機給我一張畫"):
            label = random.randint(0,829)
            label = getKey(label)
            Image = getImage(label)
            image_message = ImageSendMessage(
                original_content_url=Image[6],
                preview_image_url=Image[6]
            )
            line_bot_api.reply_message(event.reply_token,[image_message,TextSendMessage(text="畫作名稱:\n"+Image[1]),TextSendMessage(text=Image[3]),TextSendMessage(text="連結: \n"+Image[2])])
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=event.message.text))

if __name__ == "__main__":
    app.run()


