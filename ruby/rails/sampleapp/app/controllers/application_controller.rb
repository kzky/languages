class ApplicationController < ActionController::Base
  protect_from_forgery ## protect CSRF (cross site request forgery)

  rescue_from ActiveRecord::RecordNotFound, :with => :error_404
  rescue_from ActionController::UnknownAction, :with => :error_404
  rescue_from ActionController::RoutingError, :with => :error_404
  def error_404
    #render :file => '#{Rails.root}/public/404.html', :status => 404
    render :template => 'shared/error_404', :status => 404
  end
end
