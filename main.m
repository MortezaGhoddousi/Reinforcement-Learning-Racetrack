clc; clear; close all;

% load RaceTrack Environment
load envRace.mat

[s1, s2] = size(envRace);
states = zeros(size(envRace));

statecounter = 1;
s0counter = 1;
pi = 0.9;
gamma = 0.9; % Discount factor
alpha = 0.1; % Learning rate

for i = 1:s1
    for j = 1:s2
        if envRace(i, j) ~= 0
            states(i, j) = statecounter; % whole states
            statecounter = statecounter+1;
        end
        if envRace(i, j) == 2
            s0(s0counter) = states(i, j); %#ok initial states
            s0counter = s0counter+1;
        end
    end
end
  
% Define class
RT_RL = RaceTrack_RL;
RT_RL.envir = envRace;
RT_RL.gamma = gamma;
RT_RL.alpha = alpha;
RT_RL.initstate = s0;
RT_RL.states = states;
actionspace = RT_RL.createactionspace();
RT_RL.action_space = actionspace;
RT_RL.num_action = size(actionspace, 1);
statespace = RT_RL.createstatespace(statecounter-1, s0);
RT_RL.observation_space = statespace;
RT_RL.num_observation = size(statespace, 1);

% Q = zeros(env.num_observation, env.num_action);
load Q.mat

f = waitbar(0, 'Starting');
s = 100;
for iteration = 1:s
    Q = RT_RL.runepisode(Q, pi);
    waitbar(iteration/s, f, sprintf('Progress: %d %%', floor(iteration/s*100)));
    pause(0.1);
end
close(f)

save Q.mat

chain = RT_RL.rungreedy(Q, 0.1);

RT_RL.result(chain)



