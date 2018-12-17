% value iteration:

function [V, pi] = value_iteration(mdp, precision)

%IN: mdp, precision
%OUT: V, pi

% Recall: to obtain an estimate of the value function within accuracy of
% "precision" it suffices that one of the following conditions is met:
%   (i)  max(abs(V_new-V)) <= precision / (2*gamma/(1-gamma))
%   (ii) gamma^i * Rmax / (1-gamma) <= precision  -- with i the value
%   iteration count, and Rmax = max_{s,a,s'} | R(s,a,s') |
T = mdp.T; R = mdp.R; gamma = mdp.gamma;
%size(mdp) %[1,1] struct
%size(T) %[1,4] cell
num_state = size(R{1},1);
V = zeros(num_state,1);
V_new = zeros(num_state,1);
pi = zeros(num_state,1);
i = 0;
Rmax = -inf;
for a = 1:4
    temp = max(R{a}(:));
    if Rmax < temp
        Rmax = temp;
    end
end
for s = 1:num_state
    V_temp = zeros(4,1);
    for a = 1:4
        %T{a}(s,:) % row!
        sum(T{a}(s,:).*(R{a}(s,:)))
        V_temp(a) = sum(T{a}(s,:).*(R{a}(s,:) + gamma*reshape(V(:),1,num_state)),2);
    end
    [V_new(s),pi(s)] = max(V_temp);%
    i = i + 1;
end


while max(abs(V_new-V)) >= precision / (2*gamma/(1-gamma))&& gamma^i * Rmax / (1-gamma) >= precision
    V = V_new;
    for s = 1:num_state
        V_temp = zeros(4,1);
        for a = 1:4
            V_temp(a) = sum(T{a}(s,:).*(R{a}(s,:) + gamma*reshape(V(:),1,num_state)),2);
        end
        [V_new(s),pi(s)] = max(V_temp);%
        i = i + 1;
    end
end    

