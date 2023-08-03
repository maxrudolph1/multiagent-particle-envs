from make_env import make_env


env = make_env('simple_prey_capture')

obs = env.reset()



obs = env.step([0,0,0])


