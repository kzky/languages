class HelosController < ApplicationController
  def index
    render :text => "HelosController index method", :layout => true
  end
end
