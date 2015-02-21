class CreateGoods < ActiveRecord::Migration
  def change
    create_table :goods do |t|
      t.string :name
      t.integer :makers_id
      t.integer :price
      t.integer :star
      t.text :memo

      t.timestamps
    end
  end
end
