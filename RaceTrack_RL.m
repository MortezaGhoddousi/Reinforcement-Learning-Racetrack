classdef RaceTrack_RL
    % Initialization
    properties
        observation_space {mustBeNumeric}
        action_space {mustBeNumeric}
        num_observation {mustBeNumeric}
        num_action {mustBeNumeric}
        envir {mustBeNumeric}
        alpha {mustBeNumeric}
        gamma {mustBeNumeric}
        initstate {mustBeNumeric}
        states {mustBeNumeric}
    end
    methods
        function r = reset(obj, s0) % Reset Episode
            r0 = s0(randi(numel(s0)));
            [x, y] = find(obj.states==r0);
            r = [x, y, 0, 0];
        end
        % Softmax Action Selection
        function [action, numaction] = softmax(obj, Q, state, pi)
            numstate = findstate(obj, state);
            p = exp(Q(numstate, :)./pi)./sum(exp(Q(numstate, :)./(pi)));
            p = cumsum(p);
            p(end) = 1;
            numaction = find(rand<=p, 1, 'first');
            action = obj.action_space(numaction, :);
        end
        % Find State
        function numstate = findstate(obj, state)
            numstate = find(ismember(obj.observation_space, state, 'rows'));
        end
        % Create Action Space
        function actionspace = createactionspace(~)
            counter = 1;
            for i = -1:1
                for j = -1:1
                    actionspace(counter, :) = [i, j]; %#ok
                    counter = counter+1;
                end
            end
        end
        % Create State Space
        function statespace = createstatespace(obj, numstates, s0)
            counter = 1;
            for i = 1:numstates
                if sum(i==s0)==0
                    [x, y] = find(obj.states==i);
                    for j = 0:4
                        for k = 0:4
                            statespace(counter, :) = [x, y, j, k]; %#ok
                            counter = counter+1;
                        end
                    end
                else
                    [x, y] = find(obj.states==i);
                    statespace(counter, :) = [x, y, 0, 0];
                    counter = counter+1;
                end
            end
        end
        % Do the work of Agent
        function [new_state, reward, Terminal] = step(obj, action, state)
            velocity = [state(3), state(4)];
            velocity = [velocity(1)+action(1), velocity(2)+action(2)];
            velocity = min(velocity, 4);
            velocity = max(velocity, 0);
            numstate = obj.findstate(state);
            x = obj.observation_space(numstate, 1);
            y = obj.observation_space(numstate, 2);
            % Move Car
            x = x-velocity(1);
            y = y+velocity(2);
            % Calculate Reward and next State
            if obj.envir(x, y) == 0
                reward = -5;
                new_state = obj.reset(obj.initstate);
                Terminal = false;
            elseif obj.envir(x, y) == 1
                reward = -1;
                new_state = [x, y, velocity(1), velocity(2)];
                Terminal = false;
            elseif obj.envir(x, y) == 2
                reward = -1;
                new_state = [x, y, 0, 0];
                Terminal = false;
            elseif obj.envir(x, y) == 3 || obj.envir(x, y) == 4
                reward = 0;
                new_state = [x, y, velocity(1), velocity(2)];
                Terminal = true;
            end
        end
        % Run Episode
        function Q = runepisode(obj, Q, pi)
            state = obj.reset(obj.initstate);
            nappar = true;
            iter = 1;
            while nappar
                [action, numaction] = obj.softmax(Q, state, pi);
                [new_state, reward, Terminal] = obj.step(action, state);
                numstate = obj.findstate(state);
                chain(iter, :) = [state, numstate, numaction, reward]; %#ok
                if Terminal
                    break
                end
                iter = iter+1;
                state = new_state;
                pi = pi*0.99999;
            end
            R = 0;
            for i = 1:size(chain, 1)
                R = obj.gamma*R+chain(i, end);
                Q(chain(i, 5), chain(i, 6)) = Q(chain(i, 5), chain(i, 6)) + ...
                    obj.alpha*(R-Q(chain(i, 5), chain(i, 6)));
            end
        end
        function chain = rungreedy(obj, Q, pi)
            state = obj.reset(obj.initstate);
            nappar = true;
            iter = 1;
            while nappar
                [action, numaction] = obj.softmax(Q, state, pi);
                [new_state, reward, Terminal] = obj.step(action, state);
                numstate = obj.findstate(state);
                chain(iter, :) = [state, numstate, numaction, reward]; %#ok
                if Terminal
                    break
                end
                iter = iter+1;
                state = new_state;
            end
        end
        function [] = result(obj, chain)
            I = uint8(obj.envir)*80;
            for i = 1:size(chain, 1)
                c = 255;
                if chain(i, 7) == -5
                    c = 155;
                end
                I(chain(i, 1), chain(i, 2)) = c;
            end
            figure
            imshow (I)
            title('Race Map')
            xlabel('White squeres represent the direction of movement')
        end
    end
end
