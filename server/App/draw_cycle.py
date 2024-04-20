import sys
from Ball import Ball
from Player_v import Player_v
from Moment import Moment
from Constant import Constant
import numpy as np
from PIL import Image,ImageFont,ImageDraw
import cv2
import imageio
def draw_cycle(player,radius,court_side):
    if court_side:
        center_x = (94 - player.x) * 10
        center_y = (50 - player.y) * 10
    else:
        center_x = player.x * 10
        center_y = player.y * 10
    #center_x = np.random.normal(scale=0.5) + center_x
    #center_x = np.random.normal(scale=0.5) + center_x
    x1, y1 = center_x - radius, center_y - radius
    x2, y2 = center_x + radius, center_y + radius

    return((x1,y1,x2,y2))



def create_gif(scene,sample_size):

    game_id = scene.gameid
    event_id = scene.id
    moments = [Moment(moment) for moment in scene.moments]


    image = Image.open('/Users/yufu/Documents/Code/HoopPad/vis/half.jpg')
    draw = ImageDraw.Draw(image)
    radius = 10
    teamid = scene.off_teamid
    court_side = moments[0].ball.x > 47.0

    color = (0, 0, 0)
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=18)

    frames = []
    frame_num = 1

    for moment in moments[::sample_size]:
        frame_img = Image.open('/Users/yufu/Documents/Code/HoopPad/vis/half.jpg')

        # 创建新的ImageDraw对象
        frame_draw = ImageDraw.Draw(frame_img)
        player_num = 0
        for player in moment.players:
            if player.team.id == teamid:
                x1, y1, x2, y2 = draw_cycle(player,10,court_side )
                frame_draw.ellipse([(x1, y1), (x2, y2)], fill=(255, 0, 0))
                player_num = player_num + 1
                text = str(player_num)  # 您要写的数字
                text_bbox = draw.textbbox((x1, y1), text, font=font)

                # 提取文本的宽度和高度
                text_width = text_bbox[2] - text_bbox[0]
                #text_height = text_bbox[3] - text_bbox[1]

                # 计算文本的位置，以确保文本位于圆的中心
                text_x = x1 + radius - text_width // 2
                #text_y = y1 + radius - text_height // 2

                frame_draw.text((text_x, y1), text, fill=color, font=font)
            else:
                x1, y1, x2, y2 = draw_cycle(player,10,court_side )
                frame_draw.ellipse([(x1, y1), (x2, y2)], fill=(0, 0, 255))
                player_num = player_num +1
                text = str(player_num)
                text_bbox = draw.textbbox((x1, y1), text, font=font)

                # 提取文本的宽度和高度
                text_width = text_bbox[2] - text_bbox[0]
                #text_height = text_bbox[3] - text_bbox[1]

                # 计算文本的位置，以确保文本位于圆的中心
                text_x = x1 + radius - text_width // 2
                #text_y = y1 + radius - text_height // 2
                frame_draw.text((text_x, y1), text, fill=(255, 255, 255), font=font)



        ball = moment.ball
        frame_draw.ellipse(draw_cycle(ball,10,court_side ), fill=(0, 255, 0))

        #标记第几帧
        text = 'frame:'+ str(frame_num)
        text_bbox = draw.textbbox((300, 25), text, font=font)
        frame_draw.text((300, 25), text, fill=color, font=font)

        frames.append(frame_img.copy())
        frame_num +=1

    #save_name = play_name + '/'+ str(game_id) + '_' + str(event_id) + '_'+ str(frame_num) + '.mp4'
    save_name = str(game_id) + '_' + str(event_id) + '_'+ str(frame_num) + '.mp4'
    print(save_name)
    #frames[0].save(save_name, format='GIF', append_images=frames[1:], save_all=True, duration=0.2, loop=0)
    fps = 15  # 帧率
    imageio.mimsave(save_name, frames, format='mp4', fps=fps)
    frame_img.close()