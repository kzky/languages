class Maker < ActiveRecord::Base
  attr_accessible :memo, :name, :site
  has_one :good
end
