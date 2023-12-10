classdef MLP
    properties
        n_neurons
        order
        step
        alpha_init
        w_init
        Wopt
        alpha_opt
        X
        batch_size
    end
    methods
        function obj = MLP(params)
            obj.n_neurons = params.n_neurons;
            obj.order = params.order;
            obj.step = params.step;
            obj.alpha_init = params.alpha_init;
            obj.w_init = params.w_init;
            obj.Wopt = obj.initializeWeights();
            obj.alpha_opt = obj.initializeAlpha();
            obj.X = zeros(obj.order, 1);
            obj.batch_size=params.batch_size;
        end
        
        function W = initializeWeights(obj)
            if strcmp(obj.w_init, 'randn')
                W = randn(obj.order, obj.n_neurons);
            elseif strcmp(obj.w_init, 'zeros')
                W = zeros(obj.order, obj.n_neurons);
            elseif strcmp(obj.w_init, 'uniform')
                W = rand(obj.order, obj.n_neurons);
            else
                error('Invalid weight initialization method.');
            end
        end
        
        function alpha = initializeAlpha(obj)
            if strcmp(obj.alpha_init, 'randn')
                alpha = randn(obj.n_neurons, 1);
            elseif strcmp(obj.alpha_init, 'zeros')
                alpha = zeros(obj.n_neurons, 1);
            elseif strcmp(obj.alpha_init, 'uniform')
                alpha = rand(obj.n_neurons, 1);
            else
                error('Invalid alpha initialization method.');
            end
        end
        function [eqm,obj] = train(obj,epochs,X_train, d,step_algorithm)
            N_train = length(X_train);
            e = zeros(N_train, 1);
            eqm = zeros(N_train, 1);
            alpha=zeros(obj.n_neurons,1);
            X_batch=zeros(obj.batch_size,1);
            d_batch =zeros(obj.batch_size,1);
            if obj.batch_size < N_train
                num_batches = floor(N_train / obj.batch_size);
                for batch = 1:num_batches
                    alpha(:)=obj.alpha_opt(:);
                    start_idx = (batch - 1) * obj.batch_size + 1;
                    end_idx =batch * obj.batch_size;
                    X_batch(:) = X_train(start_idx:end_idx);
                    d_batch(:) = d(start_idx:end_idx);
                    e_batch=e(start_idx:end_idx);
                    grad1=0;
                    grad2=0;
                    X_t=zeros(obj.order,1);
                    for k=1:epochs
                        X_t(:)=obj.X(:);
                    for i = 1:(end_idx - start_idx + 1)
                        X_t = [X_batch(i); X_t(1:obj.order - 1)];
                        s_batch = tanh(obj.Wopt' * X_t);
                        y_batch = tanh(obj.alpha_opt' * s_batch);
                        e_batch(i) = d_batch(i) - y_batch;  
                        psi_batch = e_batch(i) * (1 - y_batch^2);
                        grad1=grad1+psi_batch*s_batch;
                        grad2=grad2+psi_batch* kron((alpha .* (1 - tanh(obj.Wopt' * X_t).^2))', X_t);
                    end
                    e(start_idx:end_idx)=e_batch(:);
                    
                    obj.alpha_opt=obj.alpha_opt+obj.step*grad1/obj.batch_size;
                    obj.Wopt=obj.Wopt+obj.step*grad2/obj.batch_size;
                    end
                    obj.X(:)=X_t(:);
                end
                if end_idx < N_train
                    alpha=obj.alpha_opt(:); 
                    remaining_data = N_train - end_idx;
                    X_remaining = zeros(remaining_data,1);
                    d_remaining = zeros(remaining_data,1);
                    X_remaining(:)= X_train(end_idx+1:end);
                    d_remaining(:) = d(end_idx+1:end);
                    e_remaining = e(end_idx+1:end);
                    for k=1:epochs
                        X_t(:)=obj.X(:);
                        for i = 1:remaining_data
                            X_t = [X_remaining(i); X_t(1:obj.order - 1)];
                            s_remaining = tanh(obj.Wopt' * X_t);
                            y_remaining = tanh(obj.alpha_opt' * s_remaining);
                            e_remaining(i) = d_remaining(i) - y_remaining;
                            psi_remaining = e_remaining(i) * (1 - y_remaining^2);
                            grad1=grad1+psi_remaining*s_remaining;
                            grad2=grad2+psi_remaining* kron((alpha .* (1 - tanh(obj.Wopt' * X_t).^2))', obj.X);
                        end
              
                        obj.alpha_opt=obj.alpha_opt+obj.step *grad1/remaining_data ;
                        obj.Wopt=obj.Wopt+obj.step*grad2/remaining_data;
                    end
                end
                e=e.^2;
                for i=1:N_train
                    eqm(i)=sum(e(1:i))/i;
                end
            
            else
                for i = 1:N_train
                    X_t=zeros(obj.order,1);
                    X_t(:)=obj.X(:);
                    for k=1:epochs
                    alpha(:)=obj.alpha_opt(:);
                    X_t = [X_train(i); X_t(1:obj.order - 1)]; 
                    s = tanh(obj.Wopt' * X_t);
                    y = tanh(obj.alpha_opt' * s);
                    e(i) = d(i) - y;
                    eqm(i) = sum(e.^2) / i;
                    psi = e(i) * (1 - y^2);
                    if strcmp(step_algorithm, 'constant')
                        eta = obj.step;
                    elseif strcmp(step_algorithm, 'diminishing')
                        eta = obj.step / sqrt(i);
                    elseif strcmp(step_algorithm, 'AdaGrad')
                        eta = obj.step / sqrt(sum(e(1:i).^2) + eps);
                    else
                        error('Invalid step size algorithm.');
                    end
                    obj.alpha_opt = obj.alpha_opt + eta * psi * s;
                    obj.Wopt = obj.Wopt + eta * psi * kron((alpha .* (1 - tanh(obj.Wopt' * X_t).^2))', X_t);
                    
                    end
                obj.X(:)=X_t(:);
                end
            end
        end    
        function predictions = predict(obj, X_test)
            N_test = length(X_test);
            predictions = zeros(N_test, 1);
            obj.X=zeros(obj.order,1);
            for i = 1:N_test
                obj.X = [X_test(i); obj.X(1:obj.order - 1)]; 
                s = tanh(obj.Wopt' * obj.X);
                y = tanh(obj.alpha_opt' * s);
                predictions(i) = y;
            end
        end
    end
end



