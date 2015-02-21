# -*- coding: utf-8 -*-
Sampleapp::Application.routes.draw do
  get "blogs/preview"
  get "blogs/reset"
  get "blogs/search"
  get "blogs/index"
  get "blogs/create"
  get "blogs/new"
  get "blogs/edit"
  get "blogs/show"
  get "blogs/update"
  get "blogs/destroy"
  get "sample/index" => "sample#index"
  get "sample/index2/:id" => "sample#index2"
  get "sample/part_template"
  get "sample/bye"
  get "sample/other_model"
  get "sample/mail"
  get "sample/pass_obj"
  get "sample/recv_obj"
  get "helos/index"
  get "info/index" #  get "info/index" => "info#index" と同じ
  get "info/post"
  get "basic_auth/index"
  post "info/post"

  
  root :to => "info#index" ## see below. public/index.html
  #match "/" => "info#index", :as => "root", :via => :get
  

  # The priority is based upon order of creation:
  # first created -> highest priority.

  # Sample of regular route:
  #   match 'products/:id' => 'catalog#view'
  # Keep in mind you can assign values other than :controller and :action

  # Sample of named route:
  #   match 'products/:id/purchase' => 'catalog#purchase', :as => :purchase
  # This route can be invoked with purchase_url(:id => product.id)

  # Sample resource route (maps HTTP verbs to controller actions automatically):
  #   resources :products

  # Sample resource route with options:
  #   resources :products do
  #     member do
  #       get 'short'
  #       post 'toggle'
  #     end
  #
  #     collection do
  #       get 'sold'
  #     end
  #   end
  
=begin
  resources :blogs do
    member do
      get 'preview'
      post 'reset'
    end

    collection do
      get 'search'
    end
  end
=end

  # Sample resource route with sub-resources:
  #   resources :products do
  #     resources :comments, :sales
  #     resource :seller
  #   end

  # Sample resource route with more complex sub-resources
  #   resources :products do
  #     resources :comments
  #     resources :sales do
  #       get 'recent', :on => :collection
  #     end
  #   end

  # Sample resource route within a namespace:
  #   namespace :admin do
  #     # Directs /admin/products/* to Admin::ProductsController
  #     # (app/controllers/admin/products_controller.rb)
  #     resources :products
  #   end

  # You can have the root of your site routed with "root"
  # just remember to delete public/index.html.
  # root :to => 'welcome#index'

  # See how all your routes lay out with "rake routes"

  # This is a legacy wild controller route that's not recommended for RESTful applications.
  # Note: This route will make all actions in every controller accessible via GET requests.
  #match ':controller(/:action(/:id))(.:format)', :controller => "application", :action => "error_404"
  #match "*path", :controller => "application", :action => "error_404"
  match "*a", :to =>  "application#error_404" # a は歴としたparameter
end
