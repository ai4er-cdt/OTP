clear all; clc;%close all; 

DoPremuteTest = 0; Do_LRP_Test = 1; CountNeurons = 0;
fn_net = '/data/Research/AABWvsACC/ECCO/MocRegression/ECCOV4r3/NN_Export_Test/OBPVsMocPsi4__NN1x5_rep1_trainbr.mat';
load(fn_net,'net','lon','bath','lat0','x','y','ValidationSamples', 'TestSamples');
[Nt,Nx] = size(x);
TVsamps = [ValidationSamples,TestSamples];

MatFold = '/data/Research/AABWvsACC/ECCO/MocRegression/ECCOV4r3/LatSpecific_XdTS_YdT2S2/Reps20/Deepcell_OBPVsMocPsi4_NN_1layers_poslin_trainbr_mapstd_divideind_dlat2_seed1//lat-60/';
nreplist = 1:20; Nneurons = 3;
alpha = 2; beta = 1;% alpha = 0.5; beta = 0.5;
LineWidth = 2; FontSize = 12;
OS = 'Linux';%'Windows';%
OutputFolder = fnos('D:/Aviv/Research/AABWvsACC/Paper_MOC_ML/Figs/',OS);

xin = x';
yp = net(xin);
xmean = mean(xin,2); xstd = std(xin,[],2); ymean = mean(y); ystd = std(y);
x_norm_all = ((xin-xmean)).*((1./xstd));
% y_norm_all = purelin(W2*poslin(W1*x_norm_all + b1)+b2);
% yout = ymean + y_norm_all*ystd;
    
%% Layerwise-Relevance Propagation (LRP) Test

if Do_LRP_Test==1
    Rinput_avg_avg = zeros([Nx,1]); Rinput_avg_avg2 = Rinput_avg_avg; Rinput_avg_avg0 = Rinput_avg_avg;
    Rinput_avg_lists = zeros([Nx,length(nreplist)]); 
    for repnum = nreplist%12; %Ensebmble average, i.e., average over results from NNs trained with different random number generator seeds
        MatFN = ['OBPVsMocPsi4__NN1x',num2str(Nneurons),'_rep',num2str(repnum),'_trainbr.mat'];
        load([MatFold,MatFN],'net');

        W1 = net.IW{1}; b1 = net.b{1}; W2 = net.LW{2}; b2 = net.b{2};
        disp(['Solution (inverse) skill: RMSE=',num2str(std(y(TVsamps)'-yp(TVsamps))./std(y)),' Sv']);

        Rinput_avg = zeros([Nx,1]);
        t_array = 1:Nt; counter = 0;
        bmax = max([0,b2]); 
        W2_p = W2; W2_p(W2_p<0) = 0; W2_n = W2; W2_n(W2_n>0) = 0;
        y_norm = purelin(W2*poslin(W1*x_norm_all + b1)+b2);
        y_norm_meanabs = mean(abs(y_norm));

        for nt=t_array %Run over all time samples
            xsamp = xin(:,nt);
            x_val = ((xsamp-xmean)).*((1./xstd));
            y_norm = purelin(W2*poslin(W1*x_val + b1)+b2);

    %         if y_norm<0.9; continue; end
    %         if y_norm>-0.3; continue; end
    %         if abs(y_norm)<0.3; continue; end

            ai = poslin(W1*x_val + b1)'; 
            if 0
                Rhidden = (ai.*W2 + bmax)./sum(ai.*W2_p );%*abs(y_norm);
                Rinput = (W1.^2./sum(sum(W1.^2)))'*Rhidden';
            elseif 0
                Rhidden_p = (ai.*W2_p)./(sum(ai.*W2_p ))*y_norm; Rhidden_p(isnan(Rhidden_p)) = 0;
                Rhidden_n = (ai.*W2_n)./(sum(ai.*W2_n ))*y_norm; Rhidden_n(isnan(Rhidden_n)) = 0;
                Rhidden = alpha*Rhidden_p - beta*Rhidden_n;
                Rinput = (W1.^2./sum(sum(W1.^2)))'*Rhidden';
            else
                yfactor = abs(y_norm)/y_norm_meanabs;% y_norm; %1; % 
                Rhidden = (ai.*W2)./(sum(ai.*W2 ))*yfactor;
    %             Rhidden = (ai.*W2)./(sum(ai.*W2 )+b2)*abs(y_norm);
                Rinput = (W1.^2./sum(sum(W1.^2)))'*Rhidden';
            end

            if isfinite(Rhidden(1))
                Rinput_avg = Rinput_avg + Rinput;
                counter = counter + 1;
            end


        end
        Rinput_avg = Rinput_avg./counter;%length(t_array);
    %     figure; yyaxis left; plot(lon,Rinput_avg); yyaxis right; plot(lon,-bath);

    %     Rinput_avg_avg = Rinput_avg_avg + Rinput_avg;
        LRP = abs(Rinput_avg);
        Rinput_avg_lists(:,repnum) = LRP;
        Rinput_avg_avg = Rinput_avg_avg + LRP;
        Rinput_avg_avg0 = Rinput_avg_avg0 + Rinput_avg;
        Rinput_avg_avg2 = Rinput_avg_avg2 + Rinput_avg.^2;
    end
    Rinput_avg_avg0 = Rinput_avg_avg0./length(nreplist);%length(t_array);
    Rinput_avg_avg = Rinput_avg_avg./length(nreplist);%length(t_array);
    Rinput_avg_avg2 = Rinput_avg_avg2./length(nreplist);%length(t_array);
    Rinput_avg_avg_std = sqrt(Rinput_avg_avg2 - Rinput_avg_avg0.^2);
    IQR = iqr(Rinput_avg_lists,2);

    fh = figure;%('Position', [50 50 1050 350]); 
    yyaxis left; plot(lon,Rinput_avg_avg,'LineWidth',LineWidth); grid on; ylabel('Relevance [0-1]');
    hold on; plot(lon,Rinput_avg_avg_std/sqrt(length(nreplist)),'k--','LineWidth',LineWidth);
    yyaxis right; plot(lon,-bath,'LineWidth',LineWidth); ylabel('Depth [m]');
    legend({'Relevance','Relevance Uncertainty','Bathymetric Depth'});
    xlim([-180,180]); xlabel('Longitude]'); 
    % hold on; plot(lon,Rinput_avg_avg2/sqrt(length(nreplist)),'k--','LineWidth',LineWidth);
    % figure; yyaxis left; plot(lon,Rinput_avg_avg2/sqrt(length(nreplist)),'LineWidth',LineWidth); yyaxis right; plot(lon,-bath); title('std');
    title('(d) Layerwise Relevance Propagation');

    set(findall(gcf,'-property','FontSize'),'FontSize',FontSize);
    figfn = [OutputFolder,'LRP_',num2str(Nneurons),'neurons.png'];
    print(fh,figfn,'-dpng','-r0'); 
    % 
    % 
    % Rhidden = W2.^2./sum(sum(W2.^2));
    % Rinput = (W1.^2./sum(sum(W1.^2)))'*Rhidden';
    % fh = figure('Position', [50 50 1050 350]); 
    % yyaxis left; plot(lon,Rinput,'LineWidth',LineWidth); grid on;
    % yyaxis right; plot(lon,-bath,'LineWidth',LineWidth);

end

%% Permutation Test

if DoPremuteTest==1
    BulkSize = 15;
    nnx = 1:BulkSize:(Nx-BulkSize);

    xmean = mean(xin,2); xstd = std(xin,[],2); ymean = mean(y); ystd = std(y);
    x_norm = ((xin-xmean)).*((1./xstd));
    
    R2_p_avg = zeros([length(nnx),1]); R2_p_avg2 = R2_p_avg;
    for repnum = nreplist%12; %Ensebmble average, i.e., average over results from NNs trained with different random number generator seeds
        MatFN = ['OBPVsMocPsi4__NN1x',num2str(Nneurons),'_rep',num2str(repnum),'_trainbr.mat']
        load([MatFold,MatFN],'net');
      
        Rinput_avg = zeros([Nx,1]);
        
        yout = net(x');
        R2 = 1-var(y(TVsamps)'-yout(TVsamps))./var(y(TVsamps)); %Coefficient of determination R^2
        disp(['Solution R2=',num2str(R2)]);

        R2_p = zeros([length(nnx),1]);
        for n=1:length(nnx)
            nx = nnx(n);
            x_p = x; a = x(:,nx:(nx+BulkSize-1)); a = a(randperm(Nt),:); x_p(:,nx:(nx+BulkSize-1)) = a; %Random permutations of a single input "feature" (i.e., specific variable and longitudinal position)
            y_perm = net(x_p');
            R2_p(n) = 1-var(y(TVsamps)'-y_perm(TVsamps))./var(y(TVsamps)); %Coefficient of determination R^2
        end
%         figure; plot(lon(nnx),R2_p); hold on; yline(R2);
        R2_p_avg = R2_p_avg + R2_p;
        R2_p_avg2 = R2_p_avg2 + R2_p.^2;

    end
    R2_p_avg = R2_p_avg./length(nreplist);%length(t_array);
    R2_p_avg2 = R2_p_avg2./length(nreplist);
    R2_p_avg_std = sqrt(R2_p_avg2 - R2_p_avg.^2);
    dR2_p_avg = R2_p_avg_std/sqrt(length(nreplist));

    lonx = lon(nnx+round(BulkSize/2));
    fh = figure;%('Position', [50 50 1050 350]); 
    plot(lonx,R2_p_avg,'k','LineWidth',LineWidth); hold on; yline(R2,'LineWidth',LineWidth);
    plot(lonx,R2_p_avg-dR2_p_avg,'k--','LineWidth',LineWidth);
    plot(lonx,R2_p_avg+dR2_p_avg,'k--','LineWidth',LineWidth);
    xlabel('Longitude'); xlim([-180,180]); grid on;
    title(['(c) Permutation test, Block Size=',num2str(BulkSize),'^{\circ} lon']); 
    
    set(findall(gcf,'-property','FontSize'),'FontSize',FontSize);
    figfn = [OutputFolder,'Permute_BlockSize',num2str(BulkSize),'deg_',num2str(Nneurons),'neurons.png'];
    print(fh,figfn,'-dpng','-r0'); 
    %     yyaxis left; plot(lon,Rinput_avg_avg,'LineWidth',LineWidth); grid on;
%     hold on; plot(lon,Rinput_avg_avg_std/sqrt(length(nreplist)),'k--','LineWidth',LineWidth);
%     yyaxis right; plot(lon,-bath,'LineWidth',LineWidth); legend({'Relevance','Relevance Uncertainty','Depth'});
%     

end




%% # of active neurons

if CountNeurons==1
   
    N_Active_Neurons = zeros([length(nreplist),1]);
    for repnum = nreplist
        MatFN = ['OBPVsMocPsi4__NN1x',num2str(Nneurons),'_rep',num2str(repnum),'_trainbr.mat']
        load([MatFold,MatFN],'net');
        W1 = net.IW{1}; W2 = net.LW{2};% b1 = net.b{1}; b2 = net.b{2};
        sum(W1.^2,2)'
%         W2
        N_Active_Neurons(repnum) = sum(W2~=0);
    end
   
end