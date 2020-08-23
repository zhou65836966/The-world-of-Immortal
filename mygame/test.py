import pygame
from pygame.color import THECOLORS
from pygame import font
from my_map import all_map
import torch
import numpy as np
import math
import random
from map_nn import map_spirit


class my_picture(object):
    def __init__(self, color_1, color_2, screen):
        self.color_1 = color_1
        self.color_2 = color_2
        self.pygame = pygame
        self.screen = screen

    def taiji(self,):
        self.pygame.draw.circle(screen, THECOLORS[self.color_1], [800, 300], 800, 0)
        self.pygame.draw.rect(screen, THECOLORS[self.color_2], [800, 0, 800, 600], 0)
        self.pygame.draw.rect(screen, THECOLORS[self.color_2], [0, 600, 800, 300], 0)
        self.pygame.draw.circle(screen, THECOLORS[self.color_1], [800, 150], 150, 0)
        self.pygame.draw.circle(screen, THECOLORS[self.color_2], [800, 450], 150, 0)
        self.pygame.draw.circle(screen, THECOLORS[self.color_2], [800, 125], 50, 0)
        self.pygame.draw.circle(screen, THECOLORS[self.color_1], [800, 475], 50, 0)
    def liangjie(self,):
        self.title = self.pygame.image.load("量劫.png")
        self.screen.blit(self.title, (740, 200))
class Button(object):
    def __init__(self, text, color, x=None, y=None, **kwargs):
        font = pygame.font.SysFont("arial", 100)
        self.screen = font.render(text, True, color)
        self.WIDTH = self.screen.get_width()
        self.HEIGHT = self.screen.get_height()
        if 'centered_x' in kwargs and kwargs['centered_x']:
            self.x = 600 // 2 - self.WIDTH // 2
        else:
            self.x = x
        if 'centered_y' in kwargs and kwargs['cenntered_y']:
            self.y = 600 // 2 - self.HEIGHT // 2
        else:
            self.y = y
    def display(self):
        screen.blit(self.screen, (self.x, self.y))
    def check_click(self, position):
        x_match = position[0] > self.x and position[0] < self.x +200
        y_match = position[1] > self.y and position[1] < self.y+100

        if x_match and y_match:
            return True
        else:
            return False
    def cl(self,position):
        if self.check_click(position) == True:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.type == 5:
                    return True
                else:
                    pass
            else:
                pass
        else:
            pass
class small_picture(object):
    def __init__(self, screen, size):
        self.pygame = pygame
        self.screen = screen
        self.size = size

        self.fog_mat = np.zeros((self.size, self.size))

    def fog_lose(self,now_index):
        a = self.fog_mat
        a[now_index[0],now_index[1]]=1
        self.fog_mat = a
        return self.fog_mat

    def display(self,):
        self.pygame.draw.rect(screen, THECOLORS['white'], [1600-2-30*self.size, 2, 30*self.size, 30*self.size], 0)
        for i in range(9):
            self.pygame.draw.line(screen, THECOLORS['black'], (1600-2-30*self.size, 2 + 30 * (i + 1)), (1598, 2 + 30 * (i + 1)), 2)
            self.pygame.draw.line(screen, THECOLORS['black'], (1600-2-30*self.size + 30 * (i + 1), 2), (1600-2-30*self.size + 30 * (i + 1), 2+30*self.size), 2)


    def small_index(self, position):
        global a_small_picture,b_small_picture
        if position[0] > 1600-2-30*self.size and position[0] < 1597 and position[1] > 2 and position[1] < 2+30*self.size-1:
            a_small_picture = int((position[0]-(1600-2-30*self.size))/30)
            b_small_picture = int((position[1]-2)/30)
            self.index_small = [a_small_picture, b_small_picture]
            font = pygame.font.SysFont("arial", 50)
            screen_0 = font.render("(%s,%s)" % (a_small_picture,b_small_picture), True, THECOLORS['white'])
            if self.fog_mat[a_small_picture, b_small_picture] == 0:
                self.pygame.draw.rect(screen, THECOLORS['black'], [1600-2-30*self.size, 3+30*self.size, 30*self.size, 60], 0)
            else:
                self.pygame.draw.rect(screen, THECOLORS['black'], [1600-2-30*self.size, 3+30*self.size, 30*self.size, 60], 0)
                screen.blit(screen_0, (1600-2-30*self.size/2, 3+30*self.size))
        else:
            self.pygame.draw.rect(screen, THECOLORS['black'], [1600-2-30*self.size, 3+30*self.size, 30*self.size, 60], 0)
    def fog(self,):
        for i in range(self.size):
            for t in range(self.size):
                if self.fog_mat[i,t] == 0:
                    self.pygame.draw.rect(screen, THECOLORS['black'],
                                          [1600 - 2 - 30 * self.size+i*30, 2 + 30 * t, 30, 30], 0)
class move_picture(object):
    def __init__(self, index, screen, lengh, space):
        left_index = (index[0]-lengh/2-math.sqrt(3)*lengh/2-space, index[1])
        self.point_left = [left_index, (math.sqrt(3)*lengh/2+left_index[0], left_index[1]+lengh/2), (math.sqrt(3)*lengh/2+left_index[0], left_index[1]-lengh/2)]
        self.point_center = [(math.sqrt(3)*lengh/2+left_index[0]+space, left_index[1]+lengh/2), (math.sqrt(3)*lengh/2+left_index[0]+space, left_index[1]-lengh/2), (math.sqrt(3)*lengh/2+lengh+left_index[0]+space, left_index[1]+lengh/2), (math.sqrt(3)*lengh/2+lengh+left_index[0]+space, left_index[1]-lengh/2)]
        self.point_right = [(math.sqrt(3)*lengh+lengh+left_index[0]+2*space, left_index[1]), (math.sqrt(3)*lengh/2+lengh+left_index[0]+2*space, left_index[1]+lengh/2), (math.sqrt(3)*lengh/2+lengh+left_index[0]+2*space, left_index[1]-lengh/2)]
        self.point_up = [(math.sqrt(3)*lengh/2+left_index[0]+space, left_index[1]+lengh/2+space),(math.sqrt(3)*lengh/2+left_index[0]+space+lengh/2, left_index[1]+lengh/2+math.sqrt(3)*lengh/2+space), (math.sqrt(3)*lengh/2+left_index[0]+space+lengh, left_index[1]+lengh/2+space)]
        self.point_down = [(math.sqrt(3)*lengh/2+left_index[0]+space, left_index[1]-lengh/2-space),(math.sqrt(3)*lengh/2+left_index[0]+space+lengh/2, left_index[1]-lengh/2-math.sqrt(3)*lengh/2-space), (math.sqrt(3)*lengh/2+left_index[0]+space+lengh, left_index[1]-lengh/2-space)]
        self.pygame = pygame
        self.screen = screen
        self.left_index = left_index
        self.lengh = lengh
        self.space = space
    def display(self,):
        self.pygame.draw.polygon(self.screen, THECOLORS['white'], self.point_left, 0)
        self.pygame.draw.polygon(self.screen, THECOLORS['white'], self.point_center, 0)
        self.pygame.draw.polygon(self.screen, THECOLORS['white'], self.point_right, 0)
        self.pygame.draw.polygon(self.screen, THECOLORS['white'], self.point_up, 0)
        self.pygame.draw.polygon(self.screen, THECOLORS['white'], self.point_down, 0)
    def check_click(self,position,x ,y):

        x_left = position[0] > self.left_index[0] and position[0] < self.left_index[0]+math.sqrt(3)*self.lengh/2
        y_left = position[1] > self.left_index[1]-(position[0]-self.left_index[0])/math.sqrt(3) and position[1] < self.left_index[1]+(position[0]-self.left_index[0])/math.sqrt(3)
        x_right = position[0] > self.left_index[0]+math.sqrt(3)*self.lengh/2+self.lengh+2*self.space and position[0] < math.sqrt(3)*self.lengh+self.lengh+self.left_index[0]+2*self.space
        y_right = position[1] > self.left_index[1] - (math.sqrt(3)*self.lengh+self.lengh+self.left_index[0]+2*self.space-position[0]) / math.sqrt(3) and position[1] < self.left_index[1] + (math.sqrt(3)*self.lengh+self.lengh+self.left_index[0]+2*self.space-position[0]) / math.sqrt(3)
        x_center = position[0] > self.point_center[0][0] and position[0] < self.point_center[2][0]
        y_center = position[1] > self.point_center[1][1] and position[1] < self.point_center[0][1]
        x_up = position[0] > self.point_up[1][0]-(self.point_up[1][1]-position[1])/math.sqrt(3) and position[0] < self.point_up[1][0]+(self.point_up[1][1]-position[1])/math.sqrt(3)
        y_up = position[1] > self.point_up[0][1] and position[1] < self.point_up[1][1]
        x_down = position[0] > self.point_up[1][0]-(position[1]-self.point_down[1][1])/math.sqrt(3) and position[0] < self.point_up[1][0]+(position[1]-self.point_down[1][1])/math.sqrt(3)
        y_down = position[1] > self.point_down[1][1] and position[1] < self.point_down[0][1]


        if x_left and y_left:
            self.pygame.draw.polygon(self.screen, THECOLORS['red'], self.point_left, 0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.type == 5:
                    if x >0:
                        x = x - 1
                        return x, y, True

        else:
            self.pygame.draw.polygon(self.screen, THECOLORS['white'], self.point_left, 0)
        if x_right and y_right:
            self.pygame.draw.polygon(self.screen, THECOLORS['red'], self.point_right, 0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.type == 5:
                    if x < 8:
                        x = x+1
                        return x, y, True
        else:
            self.pygame.draw.polygon(self.screen, THECOLORS['white'], self.point_right, 0)


        if x_up and y_up:
            self.pygame.draw.polygon(self.screen, THECOLORS['red'], self.point_up, 0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.type == 5:
                    if y < 8:
                        y = y+1
                        return x, y, True
        else:
            self.pygame.draw.polygon(self.screen, THECOLORS['white'], self.point_up, 0)
        if x_down and y_down:
            self.pygame.draw.polygon(self.screen, THECOLORS['red'], self.point_down, 0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.type == 5:
                    if y > 0:
                        y = y-1
                        return x, y, True
        else:
            self.pygame.draw.polygon(self.screen, THECOLORS['white'], self.point_down, 0)
        if x_center and y_center:
            self.pygame.draw.polygon(self.screen, THECOLORS['red'], self.point_center, 0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.type == 5:
                    return x, y, True
                else:
                    return x, y, False
            else:
                return x, y, False
        else:
            self.pygame.draw.polygon(self.screen, THECOLORS['white'], self.point_center, 0)
            return x, y, False















def begin_end(play_button,exit_button):

    if play_button.check_click(pygame.mouse.get_pos()):
        play_button = Button('origin', (255, 0, 0), 300, 700, centered_x=False)
        pygame.display.update()
    else:
        play_button = Button('origin', (255, 255, 255), 300, 700, centered_x=False)
        pygame.display.update()
    if exit_button.check_click(pygame.mouse.get_pos()):
        exit_button = Button('disillusion', (255, 0, 0), 1050, 700, centered_x=False)
        pygame.display.update()
    else:
        exit_button = Button('disillusion', (0, 0, 0), 1050, 700, centered_x=False)
        pygame.display.update()
    return play_button.display(), exit_button.display()






pygame.init()

screen = pygame.display.set_mode([1600, 900])

screen.fill(THECOLORS['white'])
pygame.display.set_caption("量劫")
cover = my_picture('white', 'black', screen)
cover.taiji()
cover.liangjie()
play_button = Button('origin', (255, 255, 255), 300, 700, centered_x=False)

exit_button = Button('disillusion', (0, 0, 0), 1050, 700, centered_x=False)
my_small_picture = small_picture(screen, 9)
move_0 = move_picture((1450, 450), screen, 50, 2)



pygame.display.flip()
mRunning = True

b = [1,1,1,10,10]
c = [0.1,0.2,0.3,0.4,0.5]
my_energy = all_map(b, 1000000, 9, c)



while mRunning:
    for event in pygame.event.get():
        play_button.display()
        exit_button.display()
        play_0, exit_0 = begin_end(play_button, exit_button)
        if play_button.cl(pygame.mouse.get_pos()):

            screen.fill(THECOLORS['black'])
            pygame.display.update()
            mRunning = False
            pygame.image.save(screen, 'FranceFlag.jpg')

        if exit_button.cl(pygame.mouse.get_pos()):
            pygame.quit()


        if event.type == pygame.QUIT:
            pygame.image.save(screen, 'FranceFlag.jpg')
            mRunning = False
        pygame.display.update()

n1 = True
x = random.randint(0, 8)
y = random.randint(0, 8)
cont = 0
yuanshen = None
load = True
while n1:
    for event in pygame.event.get():

        move_0.display()#放置移动按钮
        my_small_picture.small_index(pygame.mouse.get_pos())#查询小地图鼠标所在位置的坐标
        my_small_picture.display()#放置小地图
        pygame.draw.rect(screen, THECOLORS['red'],
                         [1600 - 2 - 30 * 9 + x * 30, 2 + 30 * y, 30, 30], 0)#标记玩家在小地图中的当前位置
        font = pygame.font.SysFont("arial", 50)#坐标的定义字体
        screen_0 = font.render("(%s,%s)" % (x, y), True, THECOLORS['white'])#生成放置鼠标所在位置的坐标文字示意A
        pygame.draw.rect(screen, THECOLORS['black'],
                              [1600 - 2 - 30 * 9, 23 + 30 * 10, 30 * 9, 60], 0)#覆盖旧文字
        screen.blit(screen_0, (1600 - 2 - 30 * 9 / 2, 23 + 30 * 10))#放置坐标文字示意A
        my_small_picture.fog_lose([x, y])#揭开地图迷雾
        x, y, next_0 = move_0.check_click(pygame.mouse.get_pos(), x, y)#检测鼠标在移动按钮中的操作
        my_small_picture.fog()#给小地图铺上迷雾
        if next_0:#如果鼠标在移动按钮上按下，使得玩家坐标移动,并且世界意志与天地灵气更新一次
            my_energy.mat_to_zero()
            pygame.draw.rect(screen, THECOLORS['black'],[50, 0, 1260, 60], 0)#覆盖旧有地图坐标中的五行灵气含量示意
            T = my_energy.inborn_energy(x, y)#得到五行灵气
            font_1 = pygame.font.SysFont("arial", 30)#定义五行灵气的字体
            screen_1 = font_1.render("(earth:%s,wood:%s,fire:%s,metal:%s,water:%s)" % (T[1], T[5], T[3], T[9], T[7]),
                                     True, THECOLORS['white'])#生成五行灵气的文字示意
            screen.blit(screen_1, (50, 0))#放置五行灵气的文字示意
            spirit = map_spirit(my_energy.earth_out,my_energy.wood_out, my_energy.fire_out, my_energy.metal_out, my_energy. water_out, 0.09, 0.001, cont, yuanshen)#将五行灵气送进世界核心
            if load == True:
                spirit.map_fact.load_state_dict(torch.load('.\world\世界意志_观测.pth'))
                spirit.map_actor.load_state_dict(torch.load('.\world\世界意志_动作.pth'))
                spirit.map_action_predict.load_state_dict(torch.load( '.\world\世界意志_实践与认知.pth'))
                spirit.map_predict.load_state_dict(torch.load( '.\world\世界意志_认知.pth'))
                spirit.map_reward.load_state_dict(torch.load( '.\world\世界意志_目标规划.pth'))
                spirit.map_critic.load_state_dict(torch.load( '.\world\世界意志_价值预估.pth'))
                spirit.map_one.load_state_dict(torch.load('.\world\世界意志_元神.pth'))
                load = False
            #世界意志输出动作，并输出当前轮次天地灵气的总情况
            actor_out, yuanshen = spirit.train()
            cont = cont + 1
            for i in range(5):
                b[i] = actor_out[i]
            for ii in range(5):
                c[i] = actor_out[i-5]





        pygame.display.update()

        if event.type == pygame.QUIT:
            pygame.image.save(screen, 'FranceFlag.jpg')
            torch.save(spirit.map_fact.state_dict(), '.\world\世界意志_观测.pth')
            torch.save(spirit.map_actor.state_dict(), '.\world\世界意志_动作.pth')
            torch.save(spirit.map_action_predict.state_dict(), '.\world\世界意志_实践与认知.pth')
            torch.save(spirit.map_predict.state_dict(), '.\world\世界意志_认知.pth')
            torch.save(spirit.map_reward.state_dict(), '.\world\世界意志_目标规划.pth')
            torch.save(spirit.map_critic.state_dict(), '.\world\世界意志_价值预估.pth')
            torch.save(spirit.map_one.state_dict(), '.\world\世界意志_元神.pth')
            n1 = False




pygame.quit()


