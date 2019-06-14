from flask import Flask, request, abort
import tensorflow as tf
import cv2
import os
import numpy as np
import time
import json
import csv
import random
from imgurpython import ImgurClient
import matplotlib.pyplot as plt

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

def judgeResolute(score):
    if(score>0.9):
        text = "我尋思相似度還是挺高的吧!"
    elif(score>0.7):
        text = "相似度普普，大概局部或顏色相似吧"
    else:
        text = "嗯......好吧，看來我的圖片庫沒有與它相似的圖，換張圖片試試?"
    return TextSendMessage(text=text)


session = {}

client_id = "bc10335ff40fd30"
client_secret = "4fbdd78d1b87a37f60d168134e90e1d6c299bf10"
access_token = "05a1cf5052f2b500708a8fd088cd66da2772a8ed"
refresh_token = "f37fdc733241eff9899345785adcf6b8c0498dc3"
client = ImgurClient(client_id, client_secret, access_token, refresh_token)

def uploadImage(url,Name):
    config = {
        'album': "Yzj1tfE",
        'name': Name,
        'title': Name,
        'description': None
    }
    image = client.upload_from_path(url, config=config, anon=False)
    return image

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

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=ImageMessage)
def handle_Image_message(event):
    time_for_name = str(int(time.time()))
    user_id = event.source.user_id
    picName = "./pictmp/"+time_for_name+".png"

    message_content = line_bot_api.get_message_content(event.message.id)
    with open(picName,"wb") as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)
    img = cv2.imdecode(np.fromfile(picName,dtype=np.uint8),-1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))

    if(user_id in session and  session[user_id] == True):
        plt_path = "./plt/"+time_for_name+".png"
        multi_predictions = sess.run(tf.nn.top_k(tf.nn.softmax(logits),3),feed_dict={x:[img],keep_prob:1})
        Images = [getImage(getKey(image)) for image in multi_predictions.indices[0]]
        Names = ["No."+getKey(image)[:-1] for image in multi_predictions.indices[0]]

        fig = plt.figure()
        plt.barh(Names[::-1],multi_predictions.values[0][::-1],height=0.7)
        plt.xlabel("Probability")
        plt.tight_layout()
        fig.savefig(plt_path)
        plt_imgur = uploadImage(plt_path,time_for_name)

        image_message = ImageSendMessage(
            original_content_url=plt_imgur["link"],
            preview_image_url=plt_imgur["link"]
        )
        arr = []
        for index,Name in enumerate(Names):
            arr.append(TextSendMessage(text=Name+"\n名稱:"+Images[index][1]+"\n連結:\n"+Images[index][2]))
        arr.insert(0,image_message)
        arr.append(judgeResolute(multi_predictions.values[0][0]))
        line_bot_api.reply_message(event.reply_token,arr)
        session[user_id] = False
        return 0

    label = sess.run(prediction,feed_dict={x:[img],keep_prob:1})
    label = getKey(label[0])

    Image = getImage(label)
    image_message = ImageSendMessage(
        original_content_url=Image[6],
        preview_image_url=Image[6]
    )
    line_bot_api.reply_message(event.reply_token,[image_message,TextSendMessage(text="畫作名稱:\n"+Image[1]),TextSendMessage(text=Image[3]),TextSendMessage(text="連結: \n"+Image[2])])
    return 0

@handler.add(MessageEvent, message=TextMessage)
def handle_Text_message(event):
    user_id = event.source.user_id
    if(event.message.text == "隨機給我一張畫"):
        session[user_id] = False
        label = random.randint(0,829)
        label = getKey(label)
        Image = getImage(label)
        image_message = ImageSendMessage(
            original_content_url=Image[6],
            preview_image_url=Image[6]
        )
        line_bot_api.reply_message(event.reply_token,[image_message,TextSendMessage(text="畫作名稱:\n"+Image[1]),TextSendMessage(text=Image[3]),TextSendMessage(text="連結: \n"+Image[2])])
        return 0
    if(event.message.text == "給我畫作其他可能結果"):
        session[user_id] = True
        line_bot_api.reply_message(event.reply_token,TextSendMessage(text="把畫給我"))
        return 0
    if("作者" in event.message.text):
        session[user_id] = False
        line_bot_api.reply_message(event.reply_token,TextSendMessage(
            text="花了一個月的時間終於寫出來啦!!\n\n 作者: Lufor129\n https://github.com/lufor129/2018_Spring_ML"))
        return 0
    if(random.random()<0.3 or ("查詢" in event.message.text) or ("功能" in event.message.text)):
        buttons_template = TemplateSendMessage(
            alt_text='請選擇服務',
            template=ButtonsTemplate(
                title='選擇服務',
                text="  ",
                thumbnail_image_url='https://imgur.com/K7WMezo.jpg',
                actions=[
                    MessageTemplateAction(
                        label="隨機給我一張畫",
                        text="隨機給我一張畫"
                    ),
                    MessageTemplateAction(
                        label="給我畫作其他可能結果",
                        text="給我畫作其他可能結果"
                    ),
                    MessageTemplateAction(
                        label="作者",
                        text="作者"
                    )
                ]
            )
        )
        line_bot_api.reply_message(event.reply_token,buttons_template)
        return 0
    else:
        session[user_id] = False
        line_bot_api.reply_message(event.reply_token,TextSendMessage(text=event.message.text))
        return 0

@handler.add(MessageEvent, message=StickerMessage)
def handle_sticker_message(event):
    sticker_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 100, 101, 102, 103, 104, 105, 106,
                   107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                   126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 401, 402]
    index_id = random.randint(0, len(sticker_ids) - 1)
    sticker_id = str(sticker_ids[index_id])
    sticker_message = StickerSendMessage(
        package_id='1',
        sticker_id=sticker_id
    )
    line_bot_api.reply_message(
        event.reply_token,
        sticker_message)


if __name__ == "__main__":
    app.run()


