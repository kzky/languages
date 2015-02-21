# -*- coding: utf-8 -*-
class InfoController < ApplicationController
  def index
    render :text => "root", :layout => true
  end

  def post # どのみちViewが呼び出される
    str = "テキストを入力。"
    str2 = params[:txt1]
    str3 = params[:txt2]
    if str2 != nil then
      str = "あなたは、「" + str2 + "&"+ str3  + "」と書きました。"
    end
    @msg = str
  end
end
