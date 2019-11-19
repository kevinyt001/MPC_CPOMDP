global w_lane  w_car l_car
load 'states_history_opp2.mat'
w_lane = 4;
w_car = 2;
l_car = 5;
X_all = states_history;

 %% right_down right_up left_up left_down
 RoadBound = [w_lane,-5*w_lane;
                w_lane,-w_lane;              
                5*w_lane,-w_lane;               
                5*w_lane,-5*w_lane;     
                
                w_lane,5*w_lane;
                w_lane, w_lane;
                5*w_lane, w_lane;
                5*w_lane,5*w_lane;
                
                -5*w_lane, 5* w_lane;
                -5*w_lane, 1* w_lane;
                -1*w_lane, 1* w_lane;
                -1*w_lane, 5* w_lane;
                
                 -5*w_lane, -5* w_lane;
                 -5*w_lane, -1* w_lane;
                  -1*w_lane, -1* w_lane;
                   -1*w_lane, -5* w_lane;
                
                ];
            
color = ['b' 'r' 'm' 'g'];
car_id = 1;        
car_opp = 2;

for i = 1: 1: size(X_all, 2)
    
    fig = figure('visible','on');
    set(gcf,'units','centimeters','position',[0,0,20,20])
    set(gca,'fontsize',20)
    
    step = i;
    pause(0.001);
    X_old = X_all(:, i);
    X_old(2) = -2;
    X_old(5) = 2;
    X_old(6) = X_old(6)-3;
    plot(RoadBound(1:4,1),RoadBound(1:4,2),'k-','LineWidth',2)
    hold on
    plot(RoadBound(5:8,1),RoadBound(5:8,2),'k-','LineWidth',2)
    plot(RoadBound(9:12,1),RoadBound(9:12,2),'k-','LineWidth',2)
    plot(RoadBound(13:16,1),RoadBound(13:16,2),'k-','LineWidth',2)
    Ego_rectangle = [X_old(1)-l_car/2*cos(X_old(3))-w_car/2*sin(X_old(3)), X_old(2)-l_car/2*sin(X_old(3))+w_car/2*cos(X_old(3));
                X_old(1)-l_car/2*cos(X_old(3))+w_car/2*sin(X_old(3)), X_old(2)-l_car/2*sin(X_old(3))-w_car/2*cos(X_old(3));
                X_old(1)+l_car/2*cos(X_old(3))-w_car/2*sin(X_old(3)), X_old(2)+l_car/2*sin(X_old(3))+w_car/2*cos(X_old(3));
                X_old(1)+l_car/2*cos(X_old(3))+w_car/2*sin(X_old(3)), X_old(2)+l_car/2*sin(X_old(3))-w_car/2*cos(X_old(3));
                X_old(1)+(l_car/2-1)*cos(X_old(3))-w_car/2*sin(X_old(3)), X_old(2)+(l_car/2-1)*sin(X_old(3))+w_car/2*cos(X_old(3));
                X_old(1)+(l_car/2-1)*cos(X_old(3))+w_car/2*sin(X_old(3)), X_old(2)+(l_car/2-1)*sin(X_old(3))-w_car/2*cos(X_old(3))];
    
   plot([Ego_rectangle(1,1) Ego_rectangle(2,1)],[Ego_rectangle(1,2) Ego_rectangle(2,2)],'-','LineWidth',2,'Color',color(car_id));
   plot([Ego_rectangle(1,1) Ego_rectangle(3,1)],[Ego_rectangle(1,2) Ego_rectangle(3,2)],'-','LineWidth',2,'Color',color(car_id));
   plot([Ego_rectangle(3,1) Ego_rectangle(4,1)],[Ego_rectangle(3,2) Ego_rectangle(4,2)],'-','LineWidth',2,'Color',color(car_id));
   plot([Ego_rectangle(2,1) Ego_rectangle(4,1)],[Ego_rectangle(2,2) Ego_rectangle(4,2)],'-','LineWidth',2,'Color',color(car_id));
   plot([Ego_rectangle(5,1) Ego_rectangle(6,1)],[Ego_rectangle(5,2) Ego_rectangle(6,2)],'-','LineWidth',2,'Color',color(car_id));         
   
   Opp_rectangle = [X_old(5)-l_car/2*cos(X_old(7))-w_car/2*sin(X_old(7)), X_old(6)-l_car/2*sin(X_old(7))+w_car/2*cos(X_old(7));
                X_old(5)-l_car/2*cos(X_old(7))+w_car/2*sin(X_old(7)), X_old(6)-l_car/2*sin(X_old(7))-w_car/2*cos(X_old(7));
                X_old(5)+l_car/2*cos(X_old(7))-w_car/2*sin(X_old(7)), X_old(6)+l_car/2*sin(X_old(7))+w_car/2*cos(X_old(7));
                X_old(5)+l_car/2*cos(X_old(7))+w_car/2*sin(X_old(7)), X_old(6)+l_car/2*sin(X_old(7))-w_car/2*cos(X_old(7));
                X_old(5)+(l_car/2-1)*cos(X_old(7))-w_car/2*sin(X_old(7)), X_old(6)+(l_car/2-1)*sin(X_old(7))+w_car/2*cos(X_old(7));
                X_old(5)+(l_car/2-1)*cos(X_old(7))+w_car/2*sin(X_old(7)), X_old(6)+(l_car/2-1)*sin(X_old(7))-w_car/2*cos(X_old(7))];
            
   plot([Opp_rectangle(1,1) Opp_rectangle(2,1)],[Opp_rectangle(1,2) Opp_rectangle(2,2)],'-','LineWidth',2,'Color',color(car_opp));
   plot([Opp_rectangle(1,1) Opp_rectangle(3,1)],[Opp_rectangle(1,2) Opp_rectangle(3,2)],'-','LineWidth',2,'Color',color(car_opp));
   plot([Opp_rectangle(3,1) Opp_rectangle(4,1)],[Opp_rectangle(3,2) Opp_rectangle(4,2)],'-','LineWidth',2,'Color',color(car_opp));
   plot([Opp_rectangle(2,1) Opp_rectangle(4,1)],[Opp_rectangle(2,2) Opp_rectangle(4,2)],'-','LineWidth',2,'Color',color(car_opp));
   plot([Opp_rectangle(5,1) Opp_rectangle(6,1)],[Opp_rectangle(5,2) Opp_rectangle(6,2)],'-','LineWidth',2,'Color',color(car_opp));         
  
   
   RoadMid = [5 0;
            20 0;
            -5 0;
            -20 0;
            0 -5;
            0 -20;
            0 5;
            0 20];


        %% Plot the midiidle line

    plot([RoadMid(1,1) RoadMid(2,1)],[RoadMid(1,2) RoadMid(2,2)],'--','color',[0.75 0.75 0.75],'LineWidth',3);
    plot([RoadMid(3,1) RoadMid(4,1)],[RoadMid(3,2) RoadMid(4,2)],'--','color',[0.75 0.75 0.75],'LineWidth',3);
    plot([RoadMid(5,1) RoadMid(6,1)],[RoadMid(5,2) RoadMid(6,2)],'--','color',[0.75 0.75 0.75],'LineWidth',3);
    plot([RoadMid(7,1) RoadMid(8,1)],[RoadMid(7,2) RoadMid(8,2)],'--','color',[0.75 0.75 0.75],'LineWidth',3);
   
   axis([-15 15 -15 15])
   axis equal
   
    set( gca, 'XTick', -15 : 5 : 15 )
    set( gca, 'YTick', -15 : 5 : 15 )
    xlabel('[m]')
    ylabel('[m]')
    annotation('textbox',...
    [0.65 0.145 0.24 0.15],...
    'String',{['v_{1} = ' num2str(X_old(4)) ' m/s '],...
               ['v_{2} = ' num2str(X_old(8)) ' m/s ']},...
    'FontSize',24,...
    'FontName','Arial',...
    'LineStyle','-',...
    'EdgeColor',[0.85 0.85 0.85],...
    'LineWidth',2,...
    'BackgroundColor',[0.95  0.95 0.95],...
    'Color','k');

    print( '-depsc', [ '-r', int2str( 1000 ) ], [ sprintf('intersection_opp_2_step_%d', i), '.eps' ] )
    
%    frame = getframe(gcf);
%     im = frame2im(frame);
%     [imind,cm] = rgb2ind(im,256);
%     outfile = 'opp_level2.gif';
%     if step == 1
%         imwrite(imind,cm,outfile,'gif','DelayTime',0,'loopcount',inf);
%     else
%         imwrite(imind,cm,outfile,'gif','DelayTime',0,'writemode','append');
%     end
    
end
close all;
