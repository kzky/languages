class CreateMakers < ActiveRecord::Migration
  def change
    create_table :makers do |t|
      t.string :name
      t.string :site
      t.text :memo

      t.timestamps
    end
  end
end
