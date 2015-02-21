class MakersController < ApplicationController
  # GET /makers
  # GET /makers.json
  def index
    @makers = Maker.all

    respond_to do |format|
      format.html # index.html.erb
      format.json { render :json => @makers }
    end
  end

  # GET /makers/1
  # GET /makers/1.json
  def show
    @maker = Maker.find(params[:id])

    respond_to do |format|
      format.html # show.html.erb
      format.json { render :json => @maker }
    end
  end

  # GET /makers/new
  # GET /makers/new.json
  def new
    @maker = Maker.new

    respond_to do |format|
      format.html # new.html.erb
      format.json { render :json => @maker }
    end
  end

  # GET /makers/1/edit
  def edit
    @maker = Maker.find(params[:id])
  end

  # POST /makers
  # POST /makers.json
  def create
    @maker = Maker.new(params[:maker])

    respond_to do |format|
      if @maker.save
        format.html { redirect_to @maker, :notice => 'Maker was successfully created.' }
        format.json { render :json => @maker, :status => :created, :location => @maker }
      else
        format.html { render :action => "new" }
        format.json { render :json => @maker.errors, :status => :unprocessable_entity }
      end
    end
  end

  # PUT /makers/1
  # PUT /makers/1.json
  def update
    @maker = Maker.find(params[:id])

    respond_to do |format|
      if @maker.update_attributes(params[:maker])
        format.html { redirect_to @maker, :notice => 'Maker was successfully updated.' }
        format.json { head :no_content }
      else
        format.html { render :action => "edit" }
        format.json { render :json => @maker.errors, :status => :unprocessable_entity }
      end
    end
  end

  # DELETE /makers/1
  # DELETE /makers/1.json
  def destroy
    @maker = Maker.find(params[:id])
    @maker.destroy

    respond_to do |format|
      format.html { redirect_to makers_url }
      format.json { head :no_content }
    end
  end
end
