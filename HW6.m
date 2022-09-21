%%
clc
clear
etha = 0.1;
etha_pi = .2;
disco = 0.9;
actions(1,1) = 0;
actions(1,2) = 1;
actions(2,1) = 1;
actions(2,2) = 0;
actions(3,1) = 0;
actions(3,2) = -1;
actions(4,2) = 0;
actions(4,1) = -1;



%%
cat = imread('cat.jpg');
cat = imresize(cat,[49 49]);
rat = imread('rat.jpg');
rat = imresize(rat,[49 49]);
cheese = imread('cheese.jpg');
cheese = imresize(cheese,[49 49]);
rat = im2double(rat);
cat = im2double(cat);
cheese = im2double(cheese);
reward = [9 4];
punish = [3 12];

path = ones(15*49+1,15*49+1,3);
for i = 1:16
    q = (i-1)*49 + 1;
    t = i*49;
    path(q,:,:) = 0;
    path(:,q,:) = 0;
    if i==1
        rat(q,:,:)=0;
        cheese(q,:,:) = 0;
        cat(q,:,:) = 0;
        rat(:,q,:)=0;
        cheese(:,q,:) = 0;
        cat(:,q,:) = 0;
    end
end
path2 = path;
path((reward(1)-1)*49+1:reward(1)*49,(reward(2)-1)*49+1:reward(2)*49,:)=cheese;
path((punish(1)-1)*49+1:punish(1)*49,(punish(2)-1)*49+1:punish(2)*49,:)=cat;




%%
values = zeros(15);
values(reward(1),reward(2)) = 100;
values(punish(1),punish(2)) = -100;
states = zeros(15);
pos_act{1,1} = [1,2];
pos_act{15,1} = [1,4];
pos_act{1,15} = [2,3];
pos_act{15,15} = [3,4];

prob_act{1,1} = [.5,.5];
prob_act{15,1} = [.5,.5];
prob_act{1,15} = [.5,.5];
prob_act{15,15} = [.5,.5];

policy_act{1,1} = exp([.5,.5])./sum(exp([.5,.5]));
policy_act{15,1} = exp([.5,.5])./sum(exp([.5,.5]));
policy_act{1,15} = exp([.5,.5])./sum(exp([.5,.5]));
policy_act{15,15} = exp([.5,.5])./sum(exp([.5,.5]));

for i=2:14
    pos_act{i,1} = [1,2,4];
    pos_act{i,15} = [2,3,4];
    pos_act{1,i} = [1,2,3];
    pos_act{15,i} = [1,3,4];
    
    prob_act{i,1} = [1/3,1/3,1/3];
    prob_act{i,15} = [1/3,1/3,1/3];
    prob_act{1,i} = [1/3,1/3,1/3];
    prob_act{15,i} = [1/3,1/3,1/3];
    
    policy_act{i,1} = exp([1/3,1/3,1/3])./sum(exp([1/3,1/3,1/3]));
    policy_act{i,15} = exp([1/3,1/3,1/3])./sum(exp([1/3,1/3,1/3]));
    policy_act{1,i} = exp([1/3,1/3,1/3])./sum(exp([1/3,1/3,1/3]));
    policy_act{15,i} = exp([1/3,1/3,1/3])./sum(exp([1/3,1/3,1/3]));
    
end
for i=2:14
    for j=2:14
        pos_act{i,j} = [1,2,3,4];
        prob_act{i,j} = [.25,.25,.25,.25];
        policy_act{i,j} = exp([.25,.25,.25,.25])./sum(exp([.25,.25,.25,.25]));
    end
end


%% TD Learning

[u,v] = gradient(states);

path3 = path2;

ch=0;
ct=0;
for j1 = 1:300
    gam0 = randi(15,1,2);
    
    if gam0 ~= reward
        flag1 = 0;
    elseif gam0 ~= punish
        flag1 = 0;
    else 
        flag1 = 1;
    end

    delt = 0;
    next_act = 0;
    path2 = path;
    c1=0;
    mov = 1;
    while flag1 == 0
        posi = gam0;
        path1 = path2;

        probab = policy_act{gam0(1),gam0(2)};
        acts1 = pos_act{gam0(1),gam0(2)};
        next_act = randsample(acts1,1,true,probab);
        
        
        

        
        
        
        gam1 = gam0 + actions(next_act,:);
        
        if gam0 == reward
            flag1=1;
            ch=ch+1;
        elseif gam0 == punish
            flag1=1;
            ct=ct+1;
        end

        delt = values(gam0(1),gam0(2))+ disco * states(gam1(1),gam1(2)) - states(gam0(1),gam0(2));
        states(gam0(1),gam0(2)) = states(gam0(1),gam0(2)) + etha*delt;
        a_i = find(next_act==acts1);
        probab(a_i) = probab(a_i) + etha_pi*delt;
        if probab(a_i)<0
            probab(a_i) = 0.001;
            ind1 = 1:length(acts1);
            ind1(ind1==a_i) = [];
            stu = sum(probab(ind1));
            edf = 0.999 - stu;
            edf = edf/3;
            probab(ind1) = probab(ind1) + edf;
        else
            probab = probab/sum(probab);
        end
        policy_act{gam0(1),gam0(2)} = probab;
        
        if gam0(1)>gam1(1)
            ga0(1) = gam0(1);
            ga1(1) = gam1(1);
        else
            ga0(1) = gam1(1);
            ga1(1) = gam0(1);
        end
        
        if gam0(2)>gam1(2)
            ga0(2) = gam0(2);
            ga1(2) = gam1(2);
        else
            ga0(2) = gam1(2);
            ga1(2) = gam0(2);
        end
        
        path2((ga1(1))*49 - 25:(ga0(1))*49 - 24,(ga1(2))*49 - 25:(ga0(2))*49 - 24,1) = 1;
        path2((ga1(1))*49 - 25:(ga0(1))*49 - 24,(ga1(2))*49 - 25:(ga0(2))*49 - 24,2:3) = 0;
        
        
        path1 = path1 +path2;
        path1((gam0(1)-1)*49+1:gam0(1)*49,(gam0(2)-1)*49+1:gam0(2)*49,:) = rat;
    
        states1 = states(15:-1:1,:);
        [u,v] = gradient(states1);
        if c1==100 || flag1==1
            subplot(1,2,1);

            imshow(path1)
            title(['Trial Number ' , num2str(j1) , '  Number of Moves ' ...
                , num2str(mov),' Cats ' num2str(ct),' Cheese ',num2str(ch)...
                ],'Interpreter','latex')

            subplot(2,2,2)
            contourf(states1)
            title(['States Values ; $${\eta}$$ = ',num2str(etha),' $${\gamma}$$ = ',num2str(disco)],'Interpreter','latex','Interpreter','latex')
            colorbar
            subplot(2,2,4)
            quiver(u,v)
            title('States Gradient','Interpreter','latex')
            pause(0.001);
            c1 = 0;
        end
        c1=c1+1;
        mov = mov+1;
        gam0 = gam1;
    end
    movs(j1) = mov;
    ct1(j1) = ct;
    ch1(j1) = ch;
end

%%
figure
plot(ct1)
hold on
plot(ch1)
legend('Cats','Cheese','Interpreter','latex')
title(['Number of Cats and Cheese ; $${\eta}$$ = ',num2str(etha),' $${\gamma}$$ = '...
    ,num2str(disco)],'Interpreter','latex','Interpreter','latex')
xlabel('Trial Number','Interpreter','latex')
grid on 
grid minor



figure
plot(ch1 - ct1)
title(['Cheese - Cats ; $${\eta}$$ = ',num2str(etha),' $${\gamma}$$ = '...
    ,num2str(disco)],'Interpreter','latex','Interpreter','latex')
xlabel('Trial Number','Interpreter','latex')
grid on 
grid minor