close all;clear all;
% World
global Dt a_hat w_hat Dx Dvx Dy u w v nx nu Cost_vec Violation_indices confidence_level n_1 n_2
Dt = 1; % delta t
a_hat = 2; % acceleration value
w_hat = 3.2; % delta y

n_1 = 2; % number of time steps for lane change
n_2 = 10; % length of lane
n_3 = 4; % number of speed values

Dx = [-n_2*a_hat*Dt^2:0.5*a_hat*Dt^2:n_2*a_hat*Dt^2];
Dvx = [-n_3*a_hat*Dt:0.5*a_hat*Dt:n_3*a_hat*Dt];
Dy = [0:w_hat/n_1:w_hat];

u = [[0;0] [a_hat;0] [-a_hat;0] [0;w_hat/(n_1*Dt)] [0;-w_hat/(n_1*Dt)]];

w = [-a_hat:a_hat:a_hat];
Pw = [0.2 0.6 0.2];
v = [[0;0] [0;0.5*a_hat*Dt] [0;-0.5*a_hat*Dt]... 
    [0.5*a_hat*Dt^2;0] [0.5*a_hat*Dt^2;0.5*a_hat*Dt] [0.5*a_hat*Dt^2;-0.5*a_hat*Dt] ...
    [-0.5*a_hat*Dt^2;0] [-0.5*a_hat*Dt^2;0.5*a_hat*Dt] [-0.5*a_hat*Dt^2;-0.5*a_hat*Dt]];
Pv = [0.6 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05];

n_Dx = length(Dx)
n_Dvx = length(Dvx)
n_Dy = length(Dy)

nx = n_Dx*n_Dvx*n_Dy
nu = 5;
nw = length(w)
ny = nx
nv = length(v(1,:))

DX = index_to_state([1:1:nx],Dx,Dvx,Dy); % Full state space values
Cost_vec = cost(DX)'; 
Violation_indices = const(DX)';

confidence_level = 0.95;

% Markov Chain
Px = zeros(nx,nx,nu); % eq 10
step = 1;
for(j=1:nx)
    for(k=1:nu)
        for(l=1:nw)
           step
           i=ffunc(j,k,l);
           Px(i,j,k) = Px(i,j,k) + Pw(l);
           step = step+1;
        end
    end
end

Py = zeros(ny,nx); % eq 11
step = 1;
for(j=1:nx)
    for(n=1:nv)
           step
           m=gfunc(j,n);
           Py(m,j) = Py(m,j) + Pv(n);
           step = step+1;
    end
end

Px_full = Px;
Px = cell(nu,1);
for(k=1:1:nu)
Px{k} = Px_full(:,:,k);
end

Rewards = zeros(nx, nu);
for i = 1:1:nx
    for k = 1:1:nu
       for j = 1:1:nx
           Rewards(i,k) = Rewards(i,k) + Px{k}(j,i)*Cost_vec(j);
       end
    end
end

Terminations = zeros(nx,1);
for i = 1:1:nx
    x_temp = DX(:,i);
    if x_temp(1) >= n_2*a_hat*Dt^2
        Terminations(i) = 1;
    end
end

% Output to Files
fileID = fopen('overtake.POMDP','w');
fprintf(fileID, 'values: rewards\n');
fprintf(fileID, 'states: %d\n', nx);
fprintf(fileID, 'actions: %d\n', nu);
fprintf(fileID, 'observations: %d\n', ny);

fprintf(fileID, '\n');
for k = 1:1:nu
%     fprintf(fileID, 'T: %d\n', k);
    for i = 1:1:nx
        for j = 1:1:nx
            if Px{k}(j,i) ~= 0
                fprintf(fileID, 'T : ');
                fprintf(fileID, '%d : %d : %d %.2f\n', [k-1, i-1, j-1, Px{k}(j, i)]);
            end
        end
%         fprintf(fileID, '\n');
    end
end

fprintf(fileID, '\n');
for i = 1:1:nx
    for j = 1:1:nx
        if Py(i,j) ~= 0
            fprintf(fileID, 'O : ');
            fprintf(fileID, '* : %d : %d %.2f\n', [i-1, j-1, Py(i, j)]);
        end
    end
    %         fprintf(fileID, '\n');
end

fprintf(fileID, '\n');
for k = 1:1:nu
    for i = 1:1:nx
        if Rewards(i,k) ~= 0
            fprintf(fileID, 'R : ');
            fprintf(fileID, '%d : %d : * : * %.2f\n', [k-1, i-1, Rewards(i, k)]);
        end
    end
    %         fprintf(fileID, '\n');
end

fprintf(fileID, '\n');
for i = 1:1:nx
    if Terminations(i) ~= 0
        fprintf(fileID, 'E : ');
        fprintf(fileID, '%d\n', i-1);
    end
end

fprintf(fileID, '\n');
for i = 1:1:size(Violation_indices,1)
    fprintf(fileID, 'V : ');
    fprintf(fileID, '%d\n', Violation_indices(i)-1);
end

fclose(fileID);

%% Helper Functions
function x_index_t1 = ffunc(x_index_t0,u_index_t0,w_index_t0)
global Dt Dx Dvx Dy u w
u_t0 = u(:,u_index_t0);
w_t0 = w(w_index_t0);
x_t0 = index_to_state(x_index_t0,Dx,Dvx,Dy);
x_t1 = [1 Dt 0;0 1 0;0 0 1]*x_t0 + [0 0;Dt 0;0 Dt]*u_t0 - [0;Dt;0]*w_t0;
x_index_t1 = state_to_index(x_t1,Dx,Dvx,Dy);
end

function y_index_t0 = gfunc(x_index_t0,v_index_t0)
global Dx Dvx Dy v
v_t0 = v(:,v_index_t0);
x_t0 = index_to_state(x_index_t0,Dx,Dvx,Dy);
y_t0 = x_t0 + [v_t0;0];
y_index_t0 = state_to_index(y_t0,Dx,Dvx,Dy);
end

function x_index = state_to_index(x,x1,x2,x3)

x1_index = dsearchn(x1',x(1,:)');
x2_index = dsearchn(x2',x(2,:)');
x3_index = dsearchn(x3',x(3,:)');

x1_length = length(x1);
x2_length = length(x2);
x3_length = length(x3);

size = [x1_length,x2_length,x3_length];
x_index = sub2ind(size,x1_index,x2_index,x3_index);
end

function x = index_to_state(x_index,x1,x2,x3)

x1_length = length(x1);
x2_length = length(x2);
x3_length = length(x3);

size = [x1_length,x2_length,x3_length];
[x1_index,x2_index,x3_index] = ind2sub(size,x_index);

x = [x1(x1_index);x2(x2_index);x3(x3_index)];
end

function cost_vec = cost(DX)
global Dt a_hat w_hat
r = 5;

cost_vec = -r/(a_hat*Dt^2)*DX(1,:)+1/w_hat*DX(3,:);
end

function violation_indices = const(DX)
global Dt a_hat w_hat n_1 n_2

% violation_indices = [];
violation_indices = find(DX(3,:)<=(n_1-1)/n_1*w_hat & abs(DX(1,:))<=n_2/2*a_hat*Dt^2);
end