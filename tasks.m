% clear all
% close all 
% clc
function [noofworkers,paymentx,total_time,truck_time,distance,truckdistance,completed,achievedtruck] = tasks( task,participants )
%function [ sum_distance,sum_bestQoI] = tasks( task, participants )
%function [ payment,total_time] = tasks( task, participants )
% sum_distance=0;
% max_time=0;
% sum_bestQoI=0;
% stand_dev_QoI=0;

R=0;
%function [sum_QoS_G_S ,sum_QoS_W_G_S, sum_cost_G_S, sum_cost_W_G_S] = tasks(task, participants)
%%
%cluster based on QOI then distance
tic
%task = xlsread('tasks_sim 70.xlsx');
%task = xlsread('tasks-HadiOtrok.xlsx');
%v=randperm(30);
%task = task1(v(1:20),:);
c=0;
n = length(task); %number of tasks

 %participants1=xlsread('participants1.xlsx');
 %participants = participants1(1:182,:); %only first 182 considered

D(:,1)=participants(1:182,1);
distance_t2t = zeros(n);
margin = 10; %to be able to see the numbers next to the circle
%%
% figure;
% clf
% title('A Cluster Based Model for QoS-OLSR Protocol')
% hold on;
% grid on;

% xlim([10000 15000]);
% ylim([10000 15000]);
%distance = zeros (n,n);
%
for i = 1:n
%        plot(task(i,2), task(i,3), 'ro');
%        text(task(i,2)+ margin, task(i,3), num2str(i));
    for j = i:n
        distance_t2t(i,j) = sqrt((task(i,2) - task(j,2))^2 + (task(i,3) - task(j,3))^2); %calculate distance between nodes
        distance_t2t(j,i) =  distance_t2t(i,j); %The matrix is symmetrical
    end %returns a 30x30 matrix of task to task distances
end
%hold off

%%
%cluster based on QOI
%eva_qoi = evalclusters(task(:,4),'kmeans','CalinskiHarabasz','KList',[1:2]);
%figure;
%plot(eva_qoi);
%idx = kmeans(task(:,4),eva_qoi.OptimalK);
idx = kmeans(task(:,4),1);

% figure;
% for i=1:n
%     plot(task(i,2), task(i,3), 'ro');
%     text(task(i,2)+ margin, task(i,3), num2str(task(i,4))); %display QOI
%     text(task(i,2)- margin, task(i,3)+ 2, num2str(task(i,1))); %display ID
%     hold on
%     grid on
% end
% hold on


%for i=1:eva_qoi.OptimalK
%

for i=1:1
    qoi_cluster{i} = task(idx == i,1:7);  %form QOI clusters vector of matrix
%     plot(task(idx == i,2),task(idx == i,3),'color',rand(1,3),'marker','o')
%     hold on
end
%title 'Cluster Assignments and Centroids based on QOI'
%hold off


%%
%Final clusters based on distance and QOI
figure;
number = 1;
%for i = 1:eva_qoi.OptimalK
for i = 1:1
    % eva = evalclusters(qoi_cluster{i}(:,2:3),'kmeans','CalinskiHarabasz','KList',[1:n/3]);
    eva = evalclusters(qoi_cluster{i}(:,2:3),'kmeans','silhouette','KList',[1*n/5:ceil(n/3)],'Distance','sqEuclidean' );
    
    if(eva.OptimalK<n/6)
        nofclusters=eva.OptimalK;
    else
        nofclusters=7;
    end
    %cluster_idx = kmeans(qoi_cluster{i}(:,2:3),nofclusters);
    cluster_idx = kmedoids(qoi_cluster{i}(:,2:3),nofclusters);
    %[~,cluster_idx] = pdist2(warehouses(:,2:3),qoi_cluster{1}(:,2:3),'euclidean','Smallest',1);
    for j=1:nofclusters %count-1
        clusters{number} = qoi_cluster{i}(cluster_idx==j,1:7);  %form QOI clusters vector of matrices
        number = number+1;
        plot(qoi_cluster{i}(cluster_idx == j,2), qoi_cluster{i}(cluster_idx == j,3), 'color',rand(1,3),'marker','o');
        %text(qoi_cluster{i}(cluster_idx == j,2), qoi_cluster{i}(cluster_idx == j,3) +100, num2str(qoi_cluster{i}(cluster_idx==j,4)));
        text(qoi_cluster{i}(cluster_idx == j,2)-100, qoi_cluster{i}(cluster_idx == j,3), num2str(qoi_cluster{i}(cluster_idx==j,1)));
        hold on
        grid on
    end
end
title 'Cluster Assignments and Centroids based on distance'


a =100;
b =200;
pickup=zeros(nofclusters,3); %array with centroids of every cluster

for j=1:nofclusters
    %r = (b-a).*rand(1,1) + a; %creating a random timewindow for pickup    
    rows=size(clusters{j},1); %finding the number of tasks in each cluster
    avg=mean(clusters{j});    
    pickuptruck=(rows+1);
    clusters{j}(pickuptruck,:)=[201,avg(1,2), avg(1,3),0,0,1,900];%setting pickup truck   
    clusters{j}([1 pickuptruck],:)=clusters{j}([pickuptruck 1],:);
    plot(avg(1,2), avg(1,3),'color',rand(1,3),'marker','o');
    pickup(j,1:3)=[j,clusters{j}(1,2:3)];    
    D(:,j+1)= sqrt(((participants(:,2) - pickup(j,2)).^2) + ((participants(:,3) - pickup(j,3)).^2) ); 
end  
%%%%%%%%%%%%%%%%%%%%%%%%
% trucks=4;
% pickup=zeros(trucks,3);
 center=mean(task);
% for j=1:trucks
%     %r = (b-a).*rand(1,1) + a; %creating a random timewindow for pickup    
%     rows=size(clusters{j},1); %finding the number of tasks in each cluster
%     avg=mean(clusters{j});    
%     pickuptruck=(rows+1);
%     clusters{j}(pickuptruck,:)=[201,avg(1,2), avg(1,3),0,0,1,900];%setting pickup truck   
%     clusters{j}([1 pickuptruck],:)=clusters{j}([pickuptruck 1],:);
%     plot(avg(1,2), avg(1,3),'color',rand(1,3),'marker','o');
%     pickup(j,1:3)=[j,clusters{j}(1,2:3)];    
%     D(:,j+1)= sqrt(((participants(:,2) - pickup(j,2)).^2) + ((participants(:,3) - pickup(j,3)).^2) ); 
% end  

% avg1=mean(pickup);

% %avg2=mean(pickup(((nofclusters/2)+1):nofclusters,:));
% for j=1:nofclusters
%     clusters{j}(1,:)=[201,avg(1,2),avg(1,3),0,0,1,5000];
% end
% 
%  plot(avg(1,2),avg(1,3),'color',rand(1,3),'marker','p');
 
hold off
%[ Truck_ST, PTruck_ST,task_index] = order_setuptime( participants, bestG, nofclusters, pickup, 2,1);
 %       truck_sol(1) = TabuSearch_plotTruck(Truck_ST, PTruck_ST, ntasks(i)-1,bestG,1, participants, task_index);
             


max_group_size = 1;
l=length (clusters); %number of clusters

bestGroups = zeros(l,max_group_size+1); %%%%%%%%%%%%%%%%%%%%%%%%

bestgroup=zeros(182,2);

workers_sel=zeros(1,n-nofclusters);
total_time=zeros(1,l);
completed=zeros(1,l);
payment=zeros(1,l);
cc=cell(1,l); %cell array similar to clusters showing the worker allocation
distance=0;
CSDdelay=0;
trucktime=zeros(3,nofclusters);
achievedtruck=0;
truckdelay=0;
counter=0;
for i=1:l
    
    flag=1; %set to 1 if you want to compute the truck measurement
    
    bestgroup(:,1)=D(:,1);
    bestgroup(:,2)=D(:,i+1);
    bestgroup=sortrows(bestgroup,2,'ascend');
    
    truckgroup(:,1)=D(:,1);
    truckgroup(:,2)=D(:,1+1);
    truckgroup=sortrows(truckgroup,2,'ascend');
    
    %budget (i) = B*length(clusters{i}(:,1));
    ntasks(i) = length(clusters{i}(:,1));
    if(ntasks(i)==1)
        
        commute=sqrt(((center(1,2) - clusters{i}(:,2)).^2) + ((center(1,3) - clusters{i}(:,3)).^2) );
        trucktime(1:2,i)=[commute,commute/7.5];
        trucktime(3,i)=trucktime(2,i)/60 +100;
        achievedtruck=achievedtruck+1;
    else 
        if(flag)
            counter=counter+1;
            bestG=reshape(truckgroup(1:2,1),1,[]);
            [ Truck_ST, PTruck_ST,task_index] = order_setuptime( participants, bestG, ntasks(i), clusters{i}, 2,1);
            truck_sol(1) = TabuSearch_plotTruck(Truck_ST, PTruck_ST, ntasks(i)-1,bestG,1, participants, task_index);
            tx=0;
            for h=1:(ntasks(i)-1)
                if(clusters{i}(h,1)~=201)
                    tx=tx+truck_sol.indcost(h)/7.5;
                    if(tx<clusters{i}(h,7))
                        achievedtruck=achievedtruck+1;
                    else
                        task_delay=(tx-clusters{i}(h,7));
                        truckdelay=truckdelay+task_delay;

                    end
                end
            end
        end
        if(counter>4)
            commute=sqrt(((center(1,2) - clusters{i}(1,2)).^2) + ((center(1,3) - clusters{i}(1,3)).^2) );
            truck_sol.Cost=truck_sol.Cost+commute;
        end
        trucktime(1:2,i)=[truck_sol.Cost,truck_sol.Cost/7.5]; %average speed for truck in m/s
        trucktime(3,i)=(trucktime(2,i)/60);%computing cost of truck in $/ minutes
        flag=0;
    end
    
    bestG=reshape(bestgroup(1:ntasks(i),1),1,[]);    
%         for k=1:n
%             for j=k+1:n
%                 T_ST(k,j) = sqrt( ((clusters(k,2) - clusters(j,2)).^2) + ((clusters(k,3) - clusters(j,3)).^2) );
%                 if(k==1)
%                     P_ST(k,j) = sqrt( ((clusters(j,2) - participants(bestG(k),2)).^2) + ((clusters(j,3) - participants(bestG(k),3)).^2) );            
%                 end
%             end
%         end
%         
% %     end
    
    %[bestG,bestQoI(i), BestGroupCost] = mainFunction1(participants, budget(i),ntasks(i), clusters{i});
    %bestGroups(i,1:length(bestG)) = bestG;
    
    %bestGroups(i,length(bestG)+1) = BestGroupCost;
    %nGroup(i)=length(find(bestG>0))-1; %size of best group
    nGroup(i)=length(bestG);
    %%
    %tabu search
   
   
   %finding the order of tasks to perform assignment in
   [ T_ST{i}, P_ST{i},task_index] = order_setuptime( participants, bestG, ntasks(i), clusters {i}, nGroup(i),0);
    final_sol{i}(1) = TabuSearch_plot(T_ST{i}, P_ST{i}, task_index, ntasks(i),bestG,1);
    order=final_sol{i}(1).Position;    
    
    x=1;
    while(x<=nGroup(i))
        if(find(bestG(1,x)==workers_sel(1,:),1))
            bestG(x)=[];
            nGroup(i)=nGroup(i)-1;
        end
        x=x+1;
    end
    
    [schedule, total_timex, paymentx, dist,nooftasks_completed,delay]=singleminded(participants,bestG, ntasks(i), clusters{i},nGroup(i),final_sol{i}(1).Position,task_index);
    
    %[schedule, total_timex, paymentx]=singleminded(participants,bestG, ntasks(i), clusters{i},nGroup(i),task_index);
    
    %ensuring same worker does not get selected for multiple clusters
    for x=2:ntasks(i)
        if~(ismember(schedule(x-1,2),workers_sel(1,:)))
        %if~(find(workers_selected(:,1)==schedule(x-1,2),1))
            workers_sel=[schedule(x-1,2),workers_sel];
            %workers_selected(end+1,1)=schedule(x-1,2);
        end
    end
    cc{i}=schedule;
    total_time(1,i)=total_timex; 
    payment(1,i)=paymentx;
    distance=distance+dist;
    CSDdelay=CSDdelay+delay;
    completed(1,i)=nooftasks_completed;
%     for k=1:nGroup(i)
%     %for k=1:1
%         final_sol{i}(k) = TabuSearch_plot(T_ST{i}, P_ST{i}, task_index, ntasks(i),bestG,k);
%         %final_sol{i}(nGroup(i)) = TabuSearch_plot(T_ST{i}, P_ST{i}, task_index, ntasks(i),bestG,k, xtasks, ytasks, participants);
%         %final_sol{i}(1:3) = TabuSearch_plot(T_ST{i}, P_ST{i}, task_index, ntasks(i),bestG,k, xtasks, ytasks, participants);
%         distance(k)=final_sol{i}(k).Cost;
%     end
%     dtemp=0;
%     %for m=1:nGroup(i)
%     for m=1:1
%         [final_cost1,~] = Cost_Function(final_sol{i}(m).Position, T_ST{i},P_ST{i}(m,:));
%      group_cost_W_GS(i) = group_cost_W_GS(i)+((final_cost1*participants(bestG(m),9))/1000);
%      dtemp=dtemp+final_cost1;
%     end
%     sum_distance(1,i)=dtemp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% calculate distance for each task
%     for m=1:ntasks(i)
%         [dtemp, QoItemp, costtemp] = Check_d_for_task(final_sol{i}, nGroup(i), distance_t2t,clusters{i}(m,1),participants, task);
%         d_for_task (clusters{i}(m,1)) = dtemp;
%         QoI_for_task (clusters{i}(m,1)) = QoItemp;
%         cost_for_task(clusters{i}(m,1)) = costtemp;
%         sum_distance=sum_distance+dtemp;
%         sum_bestQoI=sum_bestQoI+QoItemp;
%     end
    
end
%sum(payment)
completed=sum(completed);
truckdistance=truck_sol.Cost;
total_time=sum(total_time);
truck_time=sum(trucktime(2,:));
truck_cost=sum(trucktime(3,:));
noofworkers=length(nonzeros(workers_sel(workers_sel>0)));
fprintf('Total workers selected %d',length(nonzeros(workers_sel(workers_sel>0))))
fprintf('\nTotal CSD time is %f',total_time)
fprintf('\nTotal truck time is %f',truck_time)
fprintf('\nAverage CSD distance travelled is %f',distance/nooftasks_completed)
%nofclusters
fprintf('\nTotal truck distance is %f',truck_sol.Cost/nooftasks_completed)
fprintf('\nNumber of CSD completed tasks is %d',completed)
fprintf('\nNumber of truck completed tasks is %d',achievedtruck)
fprintf('\nTotal truck cost is %f',truck_cost)
fprintf('\nTotal CSD payment is %f',paymentx)
%fprintf('\nTotal CSD delay is %d',CSDdelay)
fprintf('\nTotal truck delay in minutes is %d',truckdelay/60)
end
