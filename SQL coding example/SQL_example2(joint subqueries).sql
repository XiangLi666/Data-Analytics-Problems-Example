select vip_act.*, 
       vip_cr.VIP用户当月整体购买转化率,
       vip_churn.当月新增VIP人数,
       vip_churn.前月流失VIP人数
from
    (
	select initial_vip.calendar_month as 月份,
		   count(distinct case when member_id is not null then member_id end) as VIP用户当月有预定盒子人数,
		   count(distinct vip_id) as 当月月初VIP人数,
		   round(count(distinct case when member_id is not null then member_id end)/count(distinct vip_id), 2) as VIP用户活跃度
	from
		(
		select date_format(calendar_day, '%Y-%m') as calendar_month, member_id as vip_id
		from
			(
			select date_format(calendar_day,'%Y-%m-%d') as calendar_day, member_id, sum(select_num)/sum(total) as cv 
			from
				(select distinct expect_day as calendar_day from t_order_orders
				where expect_day <= now()
				order by 1) tb1
				left join
				(select expect_day, member_id, select_num, total
				from t_order_orders
				where type = 2 and status >= 35) tb2
				on tb1.calendar_day >= tb2.expect_day
			group by 1,2    
			having cv >= 0.3 and day(calendar_day) = 1 and year(calendar_day) >= 2020
			) raw
		) initial_vip    
		left join
		(
		select date_format(expect_day, '%Y-%m') as month, member_id
		from t_order_orders
		where type = 2 and status > 0
		) customer_with_order
		on initial_vip.calendar_month = customer_with_order.month
		and initial_vip.vip_id = customer_with_order.member_id
	group by 1
	order by 1
	) vip_act
    inner join
	(
	select initial_vip.calendar_month as 月份, 
		   round(sum(select_num)/sum(total), 2) as VIP用户当月整体购买转化率
	from	
		(
		select date_format(calendar_day, '%Y-%m') as calendar_month, member_id as vip_id
		from
			(
			select date_format(calendar_day,'%Y-%m-%d') as calendar_day, member_id, sum(select_num)/sum(total) as cv 
			from
				(select distinct expect_day as calendar_day from t_order_orders
				where expect_day <= now()
				order by 1) tb1
				left join
				(select expect_day, member_id, select_num, total
				from t_order_orders
				where type = 2 and status >= 35) tb2
				on tb1.calendar_day >= tb2.expect_day
			group by 1,2    
			having cv >= 0.3 and day(calendar_day) = 1 and year(calendar_day) >= 2020
			) raw
		) initial_vip
		left join
		(
		select date_format(expect_day, '%Y-%m') as month, member_id, select_num, total
		from t_order_orders
		where type = 2 and status >= 35
		) orders
		on initial_vip.calendar_month = orders.month
		and initial_vip.vip_id = orders.member_id
	group by 1
	order by 1    
	) vip_cr
    on vip_act.月份 = vip_cr.月份
    left join
    (
	select calendar_month as 月份,
		   count(case when cur_vip_id is not null and pre_vip_id is null then 1 end) as 当月新增VIP人数,
		   count(case when cur_vip_id is null and pre_vip_id is not null then 1 end) as 前月流失VIP人数
	from
		(
		select initial_vip1.calendar_month,
			   initial_vip1.vip_id as cur_vip_id,
			   initial_vip2.vip_id as pre_vip_id
		from	
			(
			select date_format(calendar_day, '%Y-%m') as calendar_month, member_id as vip_id
			from
				(
				select date_format(calendar_day,'%Y-%m-%d') as calendar_day, member_id, sum(select_num)/sum(total) as cv 
				from
					(select distinct expect_day as calendar_day from t_order_orders
					where expect_day <= now()
					order by 1) tb1
					left join
					(select expect_day, member_id, select_num, total
					from t_order_orders
					where type = 2 and status >= 35) tb2
					on tb1.calendar_day >= tb2.expect_day
				group by 1,2    
				having cv >= 0.3 and day(calendar_day) = 1 and year(calendar_day) >= 2020
				) raw
			) initial_vip1
			left join
			(
			select date_format(calendar_day, '%Y-%m') as calendar_month, member_id as vip_id
			from
				(
				select date_format(calendar_day + interval 1 month,'%Y-%m-%d') as calendar_day, member_id, sum(select_num)/sum(total) as cv 
				from
					(select distinct expect_day as calendar_day from t_order_orders
					where expect_day <= now()
					order by 1) tb1
					left join
					(select expect_day, member_id, select_num, total
					from t_order_orders
					where type = 2 and status >= 35) tb2
					on tb1.calendar_day >= tb2.expect_day
				group by 1,2    
				having cv >= 0.3 and day(calendar_day) = 1 and date_format(calendar_day, '%Y-%m') >= '2020-02'
				) raw
			) initial_vip2
			on initial_vip1.calendar_month = initial_vip2.calendar_month
			and initial_vip1.vip_id = initial_vip2.vip_id
		union
		select initial_vip2.calendar_month,
			   initial_vip1.vip_id as cur_vip_id,
			   initial_vip2.vip_id as pre_vip_id
		from	
			(
			select date_format(calendar_day, '%Y-%m') as calendar_month, member_id as vip_id
			from
				(
				select date_format(calendar_day,'%Y-%m-%d') as calendar_day, member_id, sum(select_num)/sum(total) as cv 
				from
					(select distinct expect_day as calendar_day from t_order_orders
					where expect_day <= now()
					order by 1) tb1
					left join
					(select expect_day, member_id, select_num, total
					from t_order_orders
					where type = 2 and status >= 35) tb2
					on tb1.calendar_day >= tb2.expect_day
				group by 1,2    
				having cv >= 0.3 and day(calendar_day) = 1 and year(calendar_day) >= 2020
				) raw
			) initial_vip1
			right join
			(
			select date_format(calendar_day, '%Y-%m') as calendar_month, member_id as vip_id
			from
				(
				select date_format(calendar_day + interval 1 month,'%Y-%m-%d') as calendar_day, member_id, sum(select_num)/sum(total) as cv 
				from
					(select distinct expect_day as calendar_day from t_order_orders
					where expect_day <= now()
					order by 1) tb1
					left join
					(select expect_day, member_id, select_num, total
					from t_order_orders
					where type = 2 and status >= 35) tb2
					on tb1.calendar_day >= tb2.expect_day
				group by 1,2    
				having cv >= 0.3 and day(calendar_day) = 1 and date_format(calendar_day, '%Y-%m') >= '2020-02'
				) raw
			) initial_vip2
			on initial_vip1.calendar_month = initial_vip2.calendar_month
			and initial_vip1.vip_id = initial_vip2.vip_id
		) r
	where r.calendar_month > '2020-01'
	group by 1 
	) vip_churn 
    on vip_act.月份 = vip_churn.月份