% visualize principal components in 3 D dimension space

locat_sure_val_data = setdiff(1:length(trim_val_data),locat_unsure_val_data);
trim_val_data_sure = trim_val_data(locat_sure_val_data,1:end-1);

figure(1)
plot3(trim_val_data_sure(:,2),trim_val_data_sure(:,3),trim_val_data_sure(:,4),'bx')
hold on
plot3(trim_test_NN_dude(:,2),trim_test_NN_dude(:,3),trim_test_NN_dude(:,4),'gx')
hold on
plot3(CENTER_default(:,2),CENTER_default(:,3),CENTER_default(:,4),'r*','MarkerSize',10,'LineWidth',5)
legend('data within the clusters','data sent to ANFIS','center of clusters');
hold on
plot3(CENTER_no_default(:,2),CENTER_no_default(:,3),CENTER_no_default(:,4),'r*','MarkerSize',10,'LineWidth',5)
grid on
xlabel('Component 2');ylabel('Component 3');zlabel('Component 4');

figure(2)
plot3(trim_val_data_sure(:,1),trim_val_data_sure(:,3),trim_val_data_sure(:,4),'bx')
hold on
plot3(trim_test_NN_dude(:,1),trim_test_NN_dude(:,3),trim_test_NN_dude(:,4),'gx')
hold on
plot3(CENTER_default(:,1),CENTER_default(:,3),CENTER_default(:,4),'r*','MarkerSize',10,'LineWidth',5)
legend('data within the clusters','data sent to ANFIS','center of clusters');
hold on
plot3(CENTER_no_default(:,1),CENTER_no_default(:,3),CENTER_no_default(:,4),'r*','MarkerSize',10,'LineWidth',5)
grid on
xlabel('Component 1');ylabel('Component 3');zlabel('Component 4');

figure(3)
plot3(trim_val_data_sure(:,1),trim_val_data_sure(:,2),trim_val_data_sure(:,4),'bx')
hold on
plot3(trim_test_NN_dude(:,1),trim_test_NN_dude(:,2),trim_test_NN_dude(:,4),'gx')
hold on
plot3(CENTER_default(:,1),CENTER_default(:,2),CENTER_default(:,4),'r*','MarkerSize',10,'LineWidth',5)
legend('data within the clusters','data sent to ANFIS','center of clusters');
hold on
plot3(CENTER_no_default(:,1),CENTER_no_default(:,2),CENTER_no_default(:,4),'r*','MarkerSize',10,'LineWidth',5)
grid on
xlabel('Component 1');ylabel('Component 2');zlabel('Component 4');

figure(4)
plot3(trim_val_data_sure(:,1),trim_val_data_sure(:,2),trim_val_data_sure(:,3),'bx')
hold on
plot3(trim_test_NN_dude(:,1),trim_test_NN_dude(:,2),trim_test_NN_dude(:,3),'gx')
hold on
plot3(CENTER_default(:,1),CENTER_default(:,2),CENTER_default(:,3),'r*','MarkerSize',10,'LineWidth',5)
legend('data within the clusters','data sent to ANFIS','center of clusters');
hold on
plot3(CENTER_no_default(:,1),CENTER_no_default(:,2),CENTER_no_default(:,3),'r*','MarkerSize',10,'LineWidth',5)
grid on
xlabel('Component 1');ylabel('Component 2');zlabel('Component 3');