class BasicAuthController < ApplicationController
  http_basic_authenticate_with :name => "hoge", :password => "foo"
  def index
    
  end
end
