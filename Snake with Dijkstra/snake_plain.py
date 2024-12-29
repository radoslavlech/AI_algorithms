from turtle import Screen
from snake import Snake
from food import Food
from scoreboard import Scoreboard
import time as t

PACE = 15


width = 600
height = 600
screen = Screen()
screen.setup(width=width,height=height)
screen.bgcolor("teal")
screen.title("Snake by Radoslaw Lech")
screen.tracer(0)



my_snake = Snake()
food = Food()
scoreboard = Scoreboard()

screen.listen()
screen.onkey(my_snake.up, "Up")
screen.onkey(my_snake.down, "Down")
screen.onkey(my_snake.left, "Left")
screen.onkey(my_snake.right, "Right")



game_is_on = True
while game_is_on:
    screen.update()
    t.sleep(1/PACE)
    my_snake.move()


    #Detect collision with food
    if my_snake.head.distance(food) <15:
        food.refresh()
        my_snake.grow()
        scoreboard.increase_score()

    #Detect collision with wall
    if abs(my_snake.head.xcor())>280 or abs(my_snake.head.ycor())>280:
        game_is_on = False
        scoreboard.gameover()

    #Detect collision with tail
    for segment in my_snake.segments[1:]:
        if my_snake.head.distance(segment)<10:
            game_is_on = False
            scoreboard.gameover()


screen.exitonclick()
