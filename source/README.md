###########################################################################################
###########################################################################################

### Short Description of Each Folder

For running the code, go to the respective mainN folder you wish to run.

main0: DDQN network, state contains previous two frames + current state of the visor. Mainly built this to adjust reward function and determine where improvements could be found.

main1: DQN network. 1FC for Translation, Scaling, Rotation which equates to 18 actions. This also contains the original method without ground truth eye and shadow region detection. 

main2: Actor Critic Network using continuous action space. I just could not get this to learn properly. Loss always seems to diverge.

main3: Switched to DDQN network with value and action network separated. 3^5 actions instead of 18 actions like previously in main1. 3^5 due to each action controlling inc positive, inc negative, or inc 0 for 5 params {xpos, ypos, height, width, rotation} of visor activation map in form of a rectangle. Ground truth eye and shadow region detection is quite naive, and hardcoded as a particular color choice. 

main4: Included state of the visor activation rectangle parameters and provided it as input signal to DDQN network along with each frame. These features are concatenated at the FC layer with the encoder. Also tried to test categorical action selection (slightly greedy) with decay to greedy, but random action selection with decay to greedy still seems better.

main5: Same as main4 except we no longer do any fancy categorical action selection and strictly stick to random action selection. Environment also changed to be significantly more harder where initial starting state can be anywhere the network previously ended at.

main6: Tried completely greedy approach, where agent tries to predict the IOU for each action and make choices which always give the highest IOU. Did not work so well unfortunately.

close_form: Closed form solution to the visor project, and where the video for the market place demo is from. 



###########################################################################################
###########################################################################################

### Current best model

main5

