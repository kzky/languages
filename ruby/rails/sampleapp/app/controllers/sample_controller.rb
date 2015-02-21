# -*- coding: utf-8 -*-
class SampleController < ApplicationController
  def index
    @id = params[:id]
    @id2 = params[:id2]
    render :text => "#{@id}, #{@id2}", :layout => true
  end

  def index2
    @id = params[:id]
    render :layout => true
  end

  def part_template

  end

  def bye
    logger.debug "logger test"
    render :text => "バイバイ Rails! ", :layout => true
  end
  
  def other_model
    ## 他のモデルにアクセス可能
    ## コントローラーとモデルは一対一ではない
    @test = Test.find(1)
    
    render :text => "#{@test.name}, #{@test.price}, #{@test.author}", :layout => true
  end

  def mail
    
  end

  def pass_obj
    @obj = "test 10"
    redirect_to :controller =>"sample", :action => "recv_obj", :obj => @obj
  end
  
  def recv_obj
    logger.debug "#{p params}"
    render :text => params[:obj]
  end
  
end
